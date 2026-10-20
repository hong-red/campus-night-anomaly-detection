#!/usr/bin/env python3
import os as _os, sys as _sys
_cur = _os.path.dirname(_os.path.abspath(__file__))
for _i in range(6):
    if _os.path.exists(_os.path.join(_cur, "_paths.py")):
        _sys.path.insert(0, _cur)
        break
    _cur = _os.path.dirname(_cur)
import _paths as _PKGP
# -*- coding: utf-8 -*-
"""
FluxEV × Geolife —— 校园夜间安防异常行为检测流水线
====================================================

用途
----
把 Microsoft Geolife GPS 轨迹数据集(真实行人轨迹)作为"无人巡逻车观测到的行人轨迹"，
用 FluxEV(EVT 无监督框架) + 领域规则，输出每个时间点的【连续异常分数 0~1】与【二分类
0/1 标签】，判定依据对齐《交(1).doc》里的四类异常行为：

  1) 异常逗留 / 停留时间长 —— 停留超过 5 分钟
  2) 多人聚集               —— 深夜 >=3 人时空共现（180s 窗、50m 半径）
  3) 翻越禁区               —— 进入虚拟围栏(电子围栏)

设计要点
--------
- FluxEV 原始输出是 0/1 告警；这里把 SPOT 的 GPD 阈值映射成连续分数：
    score = clip((x - init_threshold) / (extreme_quantile - init_threshold), 0, 1)
  即：低于初始阈值 -> 0，达到告警阈值 -> 1，中间为 0~1 的"可疑度"（如 0.2 / 0.7）。
- 多维多特征融合：速度/加速度/方向变化率 各自跑 FluxEV 取 max，再与三项行为规则取 max，
  得到最终 [0,1] 异常分数，用一个阈值(默认 0.5)二值化。
- 支持两种模式：
    --mode real  在真实 Geolife 轨迹上运行，输出分数与标签（无真值）。
    --mode eval  把真实行走轨迹作为基底，注入已知的四类异常，量化评估(PR/ROC/阈值扫描)。

运行
----
    python geolife_pipeline.py --data "Geolife Trajectories 1.3/Data" --mode eval
    python geolife_pipeline.py --data "Geolife Trajectories 1.3/Data" --mode real

依赖：numpy / scipy / pandas / matplotlib / scikit-learn（无需 numba）。
"""

import os
import sys
import glob
import argparse
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体：matplotlib 默认字体不含 CJK 字形，需显式挂载系统字体
for _fp in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc"):
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False   # 避免负号显示为方块

_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FluxEV")
sys.path.insert(0, _REPO_DIR)
from spot_pipe import SPOT                      # noqa: E402
from eval_methods import adjust_predicts        # noqa: E402

warnings.filterwarnings("ignore")

# --------------------------- 常量（来自《交(1).doc》的判定标准） ---------------------------
V_STATIONARY = 0.4      # m/s，低于此视为停留
DWELL_MIN = 300         # 秒，停留超过 5 分钟 -> 异常逗留
DWELL_LOW = 120         # 秒，逗留分数开始抬升
GATHER_RADIUS = 50      # m，聚集空间半径
GATHER_DT = 180         # s，聚集时间窗
GATHER_MIN_PEOPLE = 3   # 深夜 >=3 人

# 退化保护 / 数据口径
MIN_FIT_POINTS = 60     # GPD 拟合至少需要多少个有限波动特征点
MAX_STATIC_FRAC = 0.6   # 选段后"静止点占比"上限，超过视为驻留型记录(非巡逻观测对象)

# eval 场景: 相对 test_start 的注入偏移(秒)与时长, 决定所需最短轨迹长度
EVAL_SCENARIO_END_S = 1900   # 最后一个注入段(围栏触发点)相对 test_start 的偏移
GEOFENCE_RAMP_M = 30    # 电子围栏"警戒墙"厚度(m): 越入该深度内按比例抬升分数,
                        # 避免擦到边界就立刻强报警


# --------------------------- FluxEV 核心（纯 Python，同 main.py 逻辑） ---------------------------

def calc_ewma(input_arr, alpha=0.2):
    ret = input_arr[0]
    for i in range(1, len(input_arr)):
        ret = alpha * input_arr[i] + (1 - alpha) * ret
    return ret


def calc_first_smooth(input_arr):
    return max(np.nanstd(input_arr) - np.nanstd(input_arr[:-1]), 0)


def calc_second_smooth(input_arr):
    return max(np.nanmax(input_arr) - np.nanmax(input_arr[:-1]), 0)


def fluxev_score(data_arr, train_len, period, s_w=10, p_w=5, half_d_w=2,
                 q=0.005, estimator="MOM"):
    """
    FluxEV 单变量检测，返回【连续分数 0~1】（而非 0/1 告警）。

    Returns:
        scores : (data_len - train_len,) 每个测试点的连续异常分数
        feat   : 二阶平滑后的波动特征序列(送进 SPOT 的那一维)
        th     : 动态阈值序列
        stages : dict，保留各中间阶段以便可视化:
                 pred_err(EWMA 预测误差) / fs_err(一阶平滑) /
                 fs_err_lm(局部极大) / ss_err(二阶平滑)
    """
    data_len = len(data_arr)
    spot = SPOT(q, estimator=estimator)
    d_w = half_d_w * 2
    fs_idx = s_w * 2
    fs_lm_idx = fs_idx + d_w
    ss_idx = fs_idx + half_d_w + period * (p_w - 1)

    pred_err = np.full(data_len, np.nan)
    fs_err = np.full(data_len, np.nan)
    fs_err_lm = np.full(data_len, np.nan)
    ss_err = np.full(data_len, np.nan)

    scores = np.zeros(data_len - train_len)
    ths = np.zeros(data_len - train_len)
    for i in range(s_w, data_len):
        pred_err[i] = data_arr[i] - calc_ewma(data_arr[i - s_w: i])
        if i >= fs_idx:
            fs_err[i] = calc_first_smooth(pred_err[i - s_w: i + 1])
            if i >= fs_lm_idx:
                fs_err_lm[i - half_d_w] = max(fs_err[i - d_w: i + 1])
            if i >= ss_idx:
                tem = np.append(fs_err_lm[i - period * (p_w - 1): i: period], fs_err[i])
                ss_err[i] = calc_second_smooth(tem)
        if i == train_len - 1:
            init_seg = ss_err[ss_idx: i + 1]
            init_seg = init_seg[np.isfinite(init_seg)]
            spot.fit(init_seg)
            try:
                if init_seg.size < MIN_FIT_POINTS:
                    raise ValueError("too few finite points for GPD fit")
                spot.initialize(verbose=False)
                z = spot.extreme_quantile
                # Nt<=1 时 MOM 的样本方差不可估 -> 阈值退化为 nan/inf, 会让任何波动都判满分
                if not np.isfinite(z) or spot.Nt <= 1:
                    raise ValueError("degenerate GPD fit")
            except (ZeroDivisionError, IndexError, ValueError):
                # 训练段无峰值/峰数不足 -> GPD 无法估计。此时不能伪装成"正常(0)"，
                # 否则整条轨迹静默失效；改为标记不可用，由上层降级处理。
                spot.init_threshold = float(np.nanmax(init_seg)) if init_seg.size else 0.0
                spot.extreme_quantile = np.inf
                spot.n = int(init_seg.size)
                spot.Nt = 0
                spot.init_failed = True
        if i >= train_len:
            x = ss_err[i]
            t0 = spot.init_threshold
            z = spot.extreme_quantile
            if getattr(spot, "init_failed", False):
                sc = 0.0
            elif np.isnan(x) or x <= t0:
                sc = 0.0
            elif np.isfinite(z) and z > t0:
                sc = min(max((x - t0) / (z - t0), 0.0), 1.0)
            else:
                sc = 1.0
            if getattr(spot, "init_failed", False):
                th_s, alarm_s = np.inf, 0      # 未标定: 不跑流式更新, 分数恒 0
            else:
                th_s, alarm_s = spot.run_step(x)
            if alarm_s:                       # 命中后清特征，避免污染后续点
                fs_err[i] = np.nan
                fs_err_lm[i - half_d_w] = max(fs_err[i - d_w: i + 1])
            scores[i - train_len] = sc
            ths[i - train_len] = th_s
    stages = dict(pred_err=pred_err, fs_err=fs_err, fs_err_lm=fs_err_lm, ss_err=ss_err)
    return scores, ss_err, ths, stages


# --------------------------- Geolife 数据解析 ---------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def parse_plt(path):
    """解析单个 .plt 文件 -> (lat[], lon[], t_seconds[])（相对秒）。"""
    lats, lons, times = [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                lat, lon = float(parts[0]), float(parts[1])
                date_s, time_s = parts[5].strip(), parts[6].strip()
                dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S")
            except (ValueError, IndexError):
                continue
            lats.append(lat)
            lons.append(lon)
            times.append(dt)
    if len(lats) < 20:
        return None
    lats = np.array(lats)
    lons = np.array(lons)
    t = np.array([(x - times[0]).total_seconds() for x in times])
    return lats, lons, t


def longest_continuous_segment(lats, lons, t, gap_threshold=120):
    """取时间上连续(相邻点间隔 < gap_threshold)的最长段。"""
    dt = np.diff(t)
    breaks = np.where(dt > gap_threshold)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(t)]))
    seg_lens = ends - starts
    k = int(np.argmax(seg_lens))
    return lats[starts[k]:ends[k]], lons[starts[k]:ends[k]], t[starts[k]:ends[k]]


def resample(lats, lons, t, dt_resample):
    """线性插值到等间隔时间网格，返回 (lat, lon, t_grid)。"""
    t_grid = np.arange(0, t[-1] + dt_resample, dt_resample)
    lat_grid = np.interp(t_grid, t, lats)
    lon_grid = np.interp(t_grid, t, lons)
    return lat_grid, lon_grid, t_grid


def trajectory_features(lat, lon, dt):
    """由等间隔轨迹反推 速度/加速度/方向变化率。"""
    n = len(lat)
    # 相邻点位移(米)
    d = np.array([haversine(lat[i], lon[i], lat[i + 1], lon[i + 1]) for i in range(n - 1)])
    d = np.concatenate(([0.0], d))
    # 航向
    heading = np.zeros(n)
    for i in range(1, n - 1):
        heading[i] = np.arctan2(lat[i + 1] - lat[i - 1], lon[i + 1] - lon[i - 1])
    heading[-1] = heading[-2]

    speed = d / dt
    accel = np.diff(speed, prepend=speed[0]) / dt
    dhead = np.abs(np.diff(np.unwrap(heading)))
    turn = np.concatenate(([0.0], dhead)) / dt
    turn[speed < 0.05] = 0.0
    return speed, accel, turn


def scan_walking_trajectories(data_dir, max_files=400, min_dur=900):
    """扫描 Geolife Data 目录，返回候选行走轨迹(低速、时长>15min)。"""
    plt_files = sorted(glob.glob(os.path.join(data_dir, "*", "Trajectory", "*.plt")))
    rng = np.random.default_rng(0)
    if len(plt_files) > max_files:
        plt_files = list(rng.choice(plt_files, max_files, replace=False))
    walks = []
    for p in plt_files:
        parsed = parse_plt(p)
        if parsed is None:
            continue
        lats, lons, t = parsed
        if t[-1] - t[0] < min_dur:
            continue
        lats, lons, t = longest_continuous_segment(lats, lons, t)
        if len(lats) < 60:
            continue
        lat, lon, tg = resample(lats, lons, t, 5.0)
        speed, _, _ = trajectory_features(lat, lon, 5.0)
        med = float(np.median(speed))
        p90 = float(np.percentile(speed, 90))
        frac_stat = float((speed < V_STATIONARY).mean())
        # 行走特征 + 非驻留(静止点不能占主导, 否则 dwell 会淹没整条轨迹)
        if 0.4 <= med <= 2.0 and p90 < 4.0 and frac_stat <= MAX_STATIC_FRAC:
            walks.append((len(lat), med, p, lat, lon, tg))
    walks.sort(key=lambda x: -x[0])
    return walks


# --------------------------- 行为规则分数 ---------------------------

def stationary_episode_score(speed, dt, lo, hi):
    """
    停留时长 -> 分数。speed 为等间隔序列。返回 [0,1] 分数序列。
    lo/hi: 分数从 0 线性抬升到 1 的停留时长(秒)。
    """
    n = len(speed)
    score = np.zeros(n)
    run = 0
    for i in range(n):
        if speed[i] < V_STATIONARY:
            run += dt
        else:
            run = 0
        if run <= lo:
            score[i] = 0.0
        elif run >= hi:
            score[i] = 1.0
        else:
            score[i] = (run - lo) / (hi - lo)
    return score


def geofence_score(lat, lon, zone):
    """电子围栏：zone 为 (lat0, lon0, radius_m)；在圈内 -> 1，否则 0。"""
    if zone is None:
        return np.zeros(len(lat))
    lat0, lon0, r = zone
    d = np.array([haversine(lat[i], lon[i], lat0, lon0) for i in range(len(lat))])
    return (d <= r).astype(float)


def point_in_polygon(lat, lon, poly):
    """射线法判断 (lat, lon) 是否在多边形内。poly: list[(lat, lon)]（无需闭合）。"""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]        # xi=lat, yi=lon
        xj, yj = poly[j]
        if ((yi > lon) != (yj > lon)) and \
                (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def load_geofence_polygon(path):
    """从 GeoJSON Polygon 读取电子围栏，返回 [(lat, lon), ...]（去掉闭合重复点）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("type") == "Polygon":
        ring = data["coordinates"][0]            # GeoJSON: [lon, lat]
        poly = [(pt[1], pt[0]) for pt in ring]   # -> (lat, lon)
    elif isinstance(data, dict) and isinstance(data.get("coordinates"), list):
        poly = [(pt[0], pt[1]) for pt in data["coordinates"]]
    else:
        raise ValueError(f"无法识别的围栏文件格式: {path}")
    if poly and poly[0] == poly[-1]:
        poly = poly[:-1]
    return poly


def geofence_score_polygon(lat, lon, poly):
    """多边形电子围栏：多边形内 -> 1，否则 0。"""
    if not poly:
        return np.zeros(len(lat))
    return np.array([1.0 if point_in_polygon(la, lo, poly) else 0.0
                     for la, lo in zip(lat, lon)])


def _dist_to_boundary_m(lat, lon, poly):
    """点到多边形各边的最短距离(米, 局部平面近似)。"""
    best = np.inf
    m = len(poly)
    for k in range(m):
        a = poly[k]; b = poly[(k + 1) % m]
        # 以该点为原点做局部等距投影
        la0, lo0 = lat, lon
        kx = 111320.0 * np.cos(np.radians(la0))
        ax = (a[1] - lo0) * kx; ay = (a[0] - la0) * 111320.0
        bx = (b[1] - lo0) * kx; by = (b[0] - la0) * 111320.0
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        if L2 == 0:
            d = np.hypot(ax, ay)
        else:
            tt = np.clip(-(ax * vx + ay * vy) / L2, 0.0, 1.0)
            d = np.hypot(ax + tt * vx, ay + tt * vy)
        best = min(best, d)
    return best


def geofence_score_polygon_graded(lat, lon, poly, ramp_m=GEOFENCE_RAMP_M):
    """
    多边形电子围栏(带"墙厚"的渐变判定):
      墙外            -> 0
      越入墙内不足 ramp_m -> 0~1 线性抬升(擦边轻告警)
      越入墙内 >= ramp_m -> 1(深度闯入)
    这样"碰到警戒墙"不会立刻强报警, 只有真正深入禁区内才视为越界。
    """
    if not poly:
        return np.zeros(len(lat))
    out = np.zeros(len(lat))
    for i, (la, lo) in enumerate(zip(lat, lon)):
        if not point_in_polygon(la, lo, poly):
            out[i] = 0.0
            continue
        depth = _dist_to_boundary_m(la, lo, poly)
        out[i] = min(depth / ramp_m, 1.0) if ramp_m > 0 else 1.0
    return out


def gather_score(main_lat, main_lon, others, dt, radius=GATHER_RADIUS,
                 window=GATHER_DT):
    """
    多人聚集：深夜、180s 时间窗内、50m 半径内 >=3 人共现 -> 1。
    others: list[(lat[], lon[])]，与主轨迹同一时间网格(dt)。
    时间窗：最近 window 秒内该"他人"任一时刻进入半径，即算共现（因果，不窥探未来）。
    返回 [0,1] 分数 = clip(共现他人数 / (MIN_PEOPLE-1), 0, 1)，凑满 MIN_PEOPLE 人才到 1。
    """
    n = len(main_lat)
    score = np.zeros(n)
    if not others:
        return score
    w = max(1, int(window / dt))          # 时间窗换算成步数
    for o_lat, o_lon in others:
        near = np.zeros(n, dtype=bool)
        for i in range(n):
            lo = max(0, i - w)
            d = haversine(main_lat[i], main_lon[i], o_lat[lo:i + 1], o_lon[lo:i + 1])
            if np.min(d) <= radius:
                near[i] = True
        score += near.astype(float)
    return np.clip(score / (GATHER_MIN_PEOPLE - 1), 0.0, 1.0)


# --------------------------- 融合与评估 ---------------------------

def fuse_scores(motion, dwell, gather, geof):
    return np.maximum.reduce([motion, dwell, gather, geof])


def evaluate(score, gt, test_mask):
    """在测试段上评估连续分数 -> PR/ROC/AUC/最佳阈值。"""
    from sklearn import metrics
    s = score[test_mask]
    g = gt[test_mask]
    if g.sum() == 0 or g.sum() == len(g):
        return None
    auroc = metrics.roc_auc_score(g, s)
    ap = metrics.average_precision_score(g, s)
    prec, rec, thr = metrics.precision_recall_curve(g, s)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.argmax(f1))
    best_thr = thr[k] if k < len(thr) else 0.5
    return {"auroc": auroc, "ap": ap, "best_thr": float(best_thr),
            "best_f1": float(f1[k]), "prec": prec, "rec": rec}


# --------------------------- 主流程 ---------------------------

def build_eval_scenario(walk, dt, rng):
    """
    以真实行走轨迹为基底，注入四类已知异常，返回：
        lat, lon, t, gt(组合真值), gt_dict(分行为真值), gather_others
    """
    base_lat, base_lon, t = walk
    n = len(base_lat)
    lat = base_lat.copy()
    lon = base_lon.copy()
    gt = np.zeros(n, dtype=int)
    gt_dict = {k: np.zeros(n, dtype=int) for k in ("dwell", "gather", "geofence")}

    test_start = n // 2          # FluxEV 训练段之后才注入

    def mark(kind, i0, i1):
        gt[i0:i1] = 1
        gt_dict[kind][i0:i1] = 1

    # 1) 异常逗留：6 分钟原地停留
    i0 = test_start + int(100 / dt); i1 = i0 + int(360 / dt)
    lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
    mark("dwell", i0, i1)

    # 2) 多人聚集：两个"他人"汇聚到主轨迹附近并逗留
    i0 = test_start + int(1300 / dt); i1 = i0 + int(300 / dt)
    mark("gather", i0, i1)
    others = []
    for (dlat, dlon) in ((0.0005, 0.0005), (-0.0005, 0.0004)):
        o_lat = np.full(n, lat[i0] + dlat)
        o_lon = np.full(n, lon[i0] + dlon)
        # 汇聚前远离，汇聚时贴近，之后远离
        o_lat[:i0] = lat[i0] + dlat * 8
        o_lon[:i0] = lon[i0] + dlon * 8
        o_lat[i1:] = lat[i0] + dlat * 8
        o_lon[i1:] = lon[i0] + dlon * 8
        others.append((o_lat, o_lon))

    # 4) 翻越禁区：电子围栏圆心落在行进路径上，轨迹穿过即违规
    c = min(test_start + int(1900 / dt), n - 1)
    zone = (float(lat[c]), float(lon[c]), 60.0)
    inside = geofence_score(lat, lon, zone)
    gt_dict["geofence"] = (inside > 0).astype(int)
    gt |= gt_dict["geofence"]

    return lat, lon, t, gt, gt_dict, others, zone


def run_pipeline(lat, lon, t, dt, others=None, zone=None, poly=None, q=0.005,
                 graded_geofence=True):
    """特征 -> FluxEV 连续分数 -> 行为规则 -> 融合 -> 二分类。"""
    n = len(lat)
    speed, accel, turn = trajectory_features(lat, lon, dt)

    train_len = n // 2
    period = max(1, int(60 / dt))          # 取 ~1 分钟为周期，保证 ss_idx 合理

    motion = np.zeros(n)
    feat_series = {}
    feat_stages = {}
    motion_dims_ok = 0
    for name, arr in (("speed", speed), ("accel", accel), ("turn", turn)):
        sc, feat, th, stages = fluxev_score(arr, train_len, period, q=q)
        if np.any(sc > 0):
            motion_dims_ok += 1
        motion[train_len:] = np.maximum(motion[train_len:], sc)
        feat_series[name] = (feat, th)
        feat_stages[name] = dict(stages, raw=arr, score=sc, th=th)

    dwell = stationary_episode_score(speed, dt, DWELL_LOW, DWELL_MIN)
    gather = gather_score(lat, lon, others, dt) if others else np.zeros(n)
    if poly:
        geof_bin = geofence_score_polygon(lat, lon, poly)
        geof = geofence_score_polygon_graded(lat, lon, poly) if graded_geofence else geof_bin
    else:
        geof_bin = geofence_score(lat, lon, zone)
        geof = geof_bin

    final = fuse_scores(motion, dwell, gather, geof)
    binary = (final >= 0.5).astype(int)
    return dict(speed=speed, accel=accel, turn=turn, motion=motion, dwell=dwell,
                gather=gather, geofence=geof, final=final, binary=binary,
                feat_series=feat_series, motion_degraded=(motion_dims_ok == 0),
                motion_dims_ok=motion_dims_ok, feat_stages=feat_stages,
                train_len=train_len, period=period, geofence_binary=geof_bin)


def plot_pipeline(t, lat, lon, res, gt=None, out="results/geolife.png"):
    nrows = 6 if gt is not None else 5
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 2.2 * nrows), sharex=True)

    axes[0].plot(lon, lat, lw=0.6, color="0.4")
    if gt is not None:
        axes[0].scatter(lon[gt > 0], lat[gt > 0], s=4, color="red")
    axes[0].set_title("Trajectory (red dots = ground-truth anomalies)")
    axes[0].set_ylabel("lat")

    axes[1].plot(t, res["speed"], lw=0.8, color="#0B6BFF"); axes[1].set_ylabel("Speed (m/s)")
    axes[2].plot(t, res["motion"], color="#0B6BFF", lw=1.0, label="FluxEV motion score")
    axes[2].set_ylabel("Motion score")
    axes[2].legend(fontsize=7)

    axes[3].plot(t, res["dwell"], color="#FF2D55", lw=1.0, label="dwell")
    axes[3].plot(t, res["gather"], color="#00B140", lw=1.0, label="gather")
    axes[3].plot(t, res["geofence"], color="#8A2BE2", lw=1.0, label="geofence")
    axes[3].set_ylabel("Behaviour score"); axes[3].legend(fontsize=7)

    axes[4].plot(t, res["final"], color="#E8272C", lw=1.2, label="final score")
    axes[4].axhline(0.5, color="#333333", ls="--", lw=0.9, label="threshold 0.5")
    axes[4].set_ylabel("Final score"); axes[4].legend(fontsize=7)

    if gt is not None:
        axes[5].plot(t, gt, color="#00B140", lw=1.2, drawstyle="steps-post", label="ground truth")
        axes[5].plot(t, res["binary"], color="#FF7A00", lw=1.2, drawstyle="steps-post", label="prediction (>=0.5)")
        axes[5].set_ylabel("Binary"); axes[5].legend(fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="FluxEV x Geolife campus anomaly pipeline")
    ap.add_argument("--data", type=str, default=_PKGP.GEOLIFE_DATA,
                    help="Geolife Data 目录路径")
    ap.add_argument("--mode", type=str, default="eval", choices=["real", "eval"])
    ap.add_argument("--dt", type=float, default=5.0, help="重采样间隔(s)")
    ap.add_argument("--q", type=float, default=0.005, help="SPOT 风险系数")
    ap.add_argument("--threshold", type=float, default=0.5, help="二分类阈值")
    ap.add_argument("--geofence", type=str, default=None,
                    help="电子围栏多边形文件(GeoJSON Polygon)；未指定则用演示圈")
    ap.add_argument("--max_files", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("扫描 Geolife 行走轨迹 ...")
    walks = scan_walking_trajectories(args.data, max_files=args.max_files)
    if not walks:
        print("未找到可用行走轨迹，请检查 --data 路径（应为 .../Geolife Trajectories 1.3/Data）")
        sys.exit(1)
    npts, med, path, base_lat, base_lon, t = walks[0]
    print(f"选用轨迹: {path}  (连续点数={npts}, 中位速度={med:.2f} m/s, 时长={t[-1]/60:.1f} min)")

    if args.mode == "eval":
        # 注入段从 test_start 开始按固定偏移排布, 轨迹太短会因为下标越界而崩
        # test_start = n//2, 最大下标 = test_start + EVAL_SCENARIO_END_S/dt
        need = 2 * (int(EVAL_SCENARIO_END_S / args.dt) + 2)
        if npts < need:
            print(f"[错误] eval 模式需要至少 {need} 个重采样点(约 "
                  f"{need*args.dt/60:.0f} min)，当前最长候选仅 {npts} 点。")
            print(f"       请增大 --max_files（当前 {args.max_files}）以获得更长的行走轨迹。")
            sys.exit(2)
        # 在真实轨迹上注入异常，得到真值
        lat, lon, t, gt, gt_dict, others, zone = build_eval_scenario(
            (base_lat, base_lon, t), args.dt, rng)
        res = run_pipeline(lat, lon, t, args.dt, others=others, zone=zone, q=args.q)

        test_mask = np.arange(len(lat)) >= len(lat) // 2
        ev = evaluate(res["final"], gt, test_mask)

        print("=" * 72)
        print("评估（测试段，连续分数 + 阈值二分类）")
        print("=" * 72)
        if ev:
            print(f"AUROC            = {ev['auroc']:.3f}")
            print(f"AveragePrecision = {ev['ap']:.3f}")
            print(f"最佳阈值         = {ev['best_thr']:.3f}  (最佳 F1 = {ev['best_f1']:.3f})")

        # 固定阈值 0.5 的指标
        pred = res["binary"][test_mask]
        g = gt[test_mask]
        from sklearn import metrics
        p, r, f1, _ = metrics.precision_recall_fscore_support(g, pred, average="binary")
        print(f"\n@阈值 0.5 -> Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

        # 分行为命中情况
        print("-" * 72)
        print(f"{'行为':<10}{'真值段长(步)':>12}{'命中':>6}{'首报延迟(s)':>12}")
        names = {"dwell": "异常逗留", "gather": "多人聚集", "geofence": "翻越禁区"}
        for kind, nm in names.items():
            gk = gt_dict[kind][test_mask]
            if gk.sum() == 0:
                continue
            onset = int(np.argmax(gk > 0))
            offset = int(np.where(gk > 0)[0][-1]) + 1
            pk = pred[onset:offset]
            hit = pk.sum() > 0
            lat_s = (int(np.argmax(pk > 0)) * args.dt) if hit else None
            print(f"{nm:<10}{offset-onset:>12}{str(hit):>6}{str(lat_s):>12}")

        # 阈值扫描曲线
        fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
        if ev:
            ax2[0].plot(ev["rec"], ev["prec"], label=f"AP={ev['ap']:.3f}")
            ax2[0].set_xlabel("recall"); ax2[0].set_ylabel("precision")
            ax2[0].set_title("PR curve"); ax2[0].legend()
            fpr, tpr, _ = metrics.roc_curve(g, res["final"][test_mask])
            ax2[1].plot(fpr, tpr, label=f"AUROC={ev['auroc']:.3f}")
            ax2[1].plot([0, 1], [0, 1], "k--", lw=0.6)
            ax2[1].set_xlabel("FPR"); ax2[1].set_ylabel("TPR")
            ax2[1].set_title("ROC"); ax2[1].legend()
        fig2.tight_layout()
        os.makedirs("results", exist_ok=True)
        fig2.savefig("results/geolife_curves.png", dpi=130)
        plt.close(fig2)

        plot_pipeline(t, lat, lon, res, gt=gt, out="results/geolife_eval.png")

        # 输出 CSV（含连续分数）
        df = pd.DataFrame({
            "t_s": np.round(t, 1),
            "speed": res["speed"], "accel": res["accel"], "turn": res["turn"],
            "motion_score": res["motion"], "dwell_score": res["dwell"],
            "gather_score": res["gather"],
            "geofence_score": res["geofence"], "final_score": res["final"],
            "binary_label": res["binary"], "gt_label": gt,
        })
        df.to_csv("results/geolife_scores.csv", index=False)

    else:   # real 模式
        lat, lon, t = base_lat, base_lon, t
        poly = load_geofence_polygon(args.geofence) if args.geofence else None
        if poly:
            zone = None
            print(f"电子围栏: {args.geofence} ({len(poly)} 个顶点)")
        else:
            # 电子围栏需由地图/虚拟围栏配置，这里取轨迹中段一个点作演示圈
            c = len(lat) * 2 // 3
            zone = (float(lat[c]), float(lon[c]), 60.0)
        res = run_pipeline(lat, lon, t, args.dt, others=None, zone=zone,
                           poly=poly, q=args.q)
        plot_pipeline(t, lat, lon, res, gt=None, out="results/geolife_real.png")
        df = pd.DataFrame({
            "t_s": np.round(t, 1), "speed": res["speed"],
            "motion_score": res["motion"], "dwell_score": res["dwell"],
            "gather_score": res["gather"],
            "geofence_score": res["geofence"], "final_score": res["final"],
            "binary_label": res["binary"],
        })
        df.to_csv("results/geolife_real_scores.csv", index=False)
        n_alarm = int(df["binary_label"].sum())
        n_geo = int((res["geofence"] > 0).sum())
        frac_stat = float((res["speed"] < V_STATIONARY).mean())
        print(f"真实轨迹处理完成，共 {len(df)} 个时间点，异常(分数>=0.5)点数 = {n_alarm}")
        print(f"电子围栏命中点数 = {n_geo}")
        print(f"静止点占比 = {frac_stat*100:.1f}%"
              + ("  [警告] 超过 %.0f%%，属驻留型记录，异常率不可作巡逻口径解读"
                 % (MAX_STATIC_FRAC*100) if frac_stat > MAX_STATIC_FRAC else ""))
        if res.get("motion_degraded"):
            print("[警告] 运动维度 GPD 拟合退化(训练段波动峰数不足)，motion 分数已置 0，"
                  "本次异常仅来自行为规则")
        print("final_score 分位数:", np.round(np.percentile(res["final"], [50, 90, 95, 99]), 3))

    print("-" * 72)
    print("输出: results/ 目录下的 CSV(分数) 与 PNG(图)")


if __name__ == "__main__":
    main()
