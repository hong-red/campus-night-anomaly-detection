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
SDD(Stanford Drone Dataset) 共用工具 —— 米制 FluxEV 规则管线 + TCN 训练/打分
============================================================================

把 F:\\仿真跑 的两个方案复用到斯坦福无人机数据集：
  - 方案1 规则/EVT：复用 geolife_pipeline 的 FluxEV-SPOT 核心(纯 Python)，坐标由
    "像素 -> 米"(行人 bbox 主轴 ~1.7m 身高估 m/px)；
  - 方案2 深度学习：复用 tcn_dl_branch 的 TCN 轨迹预测分支(同架构、同自监督 MSE)，
    在 SDD 正常行走轨迹上重训(dt=1s，因为 SDD 俯拍剪辑短、轨迹多为 30~150s)。

规则阈值按 SDD 机载俯拍剪辑尺度重标定(判定"形式"不变，只改时长/半径量纲)：
  dwell   长时间停留: speed < 0.4 m/s 持续 > 15s(30s 封顶)
  gather  多人聚集:    >=3 人在 5m 内共现
  geofence 地理围栏:   圆形虚拟禁区(半径可配)，入内即告警

TCN 核心 API 与 tcn_branch 一致：predict / frame_ade / zclip(置信度门限)。
"""
import os
import sys
import glob
import json

# Windows 下必须先 import torch 再 import numpy/pandas(c10.dll 冲突，见 tcn_branch.py)
import torch
import numpy as np

_REPO = _PKGP.CODE_TSINGHUA
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO, os.path.join(_REPO, "tcn_dl_branch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import geolife_pipeline as gp
gp.MIN_FIT_POINTS = 15          # 放宽短轨迹的 GPD 拟合点数下限(同 run_sdd.py)
import tcn_branch as T
import run_sdd                  # parse_raw / to_tracks / estimate_scale / features_2d / FPS

# 中文字体(仅影响 docstring 打印与个别中文图例；配图仍用英文)
for _fp in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc"):
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = os.path.join(_HERE, "results")
SDD_TCN_CKPT = os.path.join(RESULTS_DIR, "sdd_tcn_checkpoint.pt")

# --------------------------- SDD 规则常量(机载俯拍尺度) ---------------------------
V_STAT = 0.4            # m/s，低于此视为停留
DWELL_LO = 15.0         # s，停留分数开始抬升(Geolife 为 120s，SDD 剪辑短故缩到 15s)
DWELL_HI = 30.0         # s，停留分数封顶
GATHER_RADIUS = 5.0     # m，聚集半径(Geolife GPS 为 50m，SDD 俯拍为 5m)
GATHER_MIN_PEOPLE = 3   # >=3 人
GATHER_WINDOW = 5.0     # s，聚集共现时间窗
ZONE_RADIUS = 12.0      # m，虚拟禁区半径

# --------------------------- 项目配色(与 F:\\仿真跑 Nature 风格一致) ---------------------------
C_RULE = "#1B7837"      # 规则(if)
C_DL = "#eb6834"        # 深度学习(TCN)
C_FUSED = "#0b0b0b"     # 融合(软+硬)
C_FLUXEV = "#9AA0A6"    # FluxEV 运动 EVT
C_WANDER = "#2a78d6"    # 徘徊
C_LOITER = "#eb6834"    # 原地逗留
C_CHANCE = "#c3c2b7"    # 随机基线
C_GT = "#e34948"        # 真值区间
C_INK = "#0b0b0b"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9.0,
    "legend.fontsize": 8.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "savefig.dpi": 220,
})


# --------------------------- SDD 数据解析(像素 -> 米) ---------------------------

def load_sdd_ped_tracks(scenes=None):
    """扫描 SDD，返回 list[(scene, x, y, t)]，x/y 为米制绝对坐标，t 为秒。

    复用 run_sdd 的 parse_raw / to_tracks / estimate_scale：
      像素坐标用"行人 bbox 长边(主轴)中位数 ≈ 1.7m 身高"估每场景 m/px。
    """
    root = _PKGP.SDD_DATA
    names = scenes or sorted([d for d in os.listdir(root)
                              if os.path.isdir(os.path.join(root, d)) and d != ".git"])
    out = []
    for sc in names:
        vids = sorted(glob.glob(os.path.join(root, sc, "video*", "annotations.txt")))
        rows = []
        for vp in vids:
            rows += run_sdd.parse_raw(vp)
        scale = run_sdd.estimate_scale(rows)
        for x, y, t, label in run_sdd.to_tracks(rows):
            if label == "Pedestrian":
                out.append((sc, x * scale, y * scale, t))
    return out


def resample_xy(x, y, t, dt):
    """线性插值到等间隔时间网格(米)。"""
    tg = np.arange(0, t[-1] + dt, dt)
    return np.interp(tg, t, x), np.interp(tg, t, y), tg


def collect_sdd_walking(dt=1.0, max_traj=None, seed=0, min_len=40):
    """收集"行走"行人轨迹(中位速度 0.4~2.0 m/s 且 p90<4.0)，重采样到 dt。

    返回 list[(scene, x, y, t)]，x/y 为米制绝对坐标。
    """
    tracks = load_sdd_ped_tracks()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tracks))
    out = []
    for k in order:
        sc, x, y, t = tracks[k]
        if t[-1] < min_len * dt:
            continue
        xg, yg, tg = resample_xy(x, y, t, dt)
        if len(xg) < min_len:
            continue
        d = np.hypot(np.diff(xg), np.diff(yg))
        speed = np.concatenate(([0.0], d)) / dt
        med = float(np.median(speed))
        p90 = float(np.percentile(speed, 90))
        if 0.4 <= med <= 2.0 and p90 < 4.0:
            out.append((sc, xg, yg, tg))
        if max_traj and len(out) >= max_traj:
            break
    return out


# --------------------------- 米制行为规则分数 ---------------------------

def stationary_episode(speed, dt, lo=DWELL_LO, hi=DWELL_HI):
    """停留时长 -> [0,1] 分数(与 Geolife 同形，阈值按 SDD 尺度)。"""
    n = len(speed)
    score = np.zeros(n)
    run = 0.0
    for i in range(n):
        run = run + dt if speed[i] < V_STAT else 0.0
        if run <= lo:
            score[i] = 0.0
        elif run >= hi:
            score[i] = 1.0
        else:
            score[i] = (run - lo) / (hi - lo)
    return score


def gather_score_2d(x, y, others, dt, radius=GATHER_RADIUS,
                    min_people=GATHER_MIN_PEOPLE, window=GATHER_WINDOW):
    """多人聚集：window 秒内、radius 米内 >= min_people 人共现 -> [0,1]。"""
    n = len(x)
    score = np.zeros(n)
    if not others:
        return score
    w = max(1, int(window / dt))
    for ox, oy in others:
        near = np.zeros(n, dtype=bool)
        for i in range(n):
            lo = max(0, i - w)
            d = np.hypot(ox[lo:i + 1] - x[i], oy[lo:i + 1] - y[i])
            if np.min(d) <= radius:
                near[i] = True
        score += near.astype(float)
    return np.clip(score / (min_people - 1), 0.0, 1.0)


def geofence_score_2d(x, y, zone):
    """圆形地理围栏：zone=(cx, cy, r)；圈内 -> 1，否则 0。"""
    if zone is None:
        return np.zeros(len(x))
    cx, cy, r = zone
    d = np.hypot(x - cx, y - cy)
    return (d <= r).astype(float)


def run_sdd_pipeline(x, y, t, dt, others=None, zone=None, q=0.005,
                     dwell_lo=DWELL_LO, dwell_hi=DWELL_HI,
                     gather_radius=GATHER_RADIUS, gather_min=GATHER_MIN_PEOPLE):
    """米制 FluxEV 运动分数 + 三项行为规则 -> 融合(与 geolife_pipeline 同构)。"""
    n = len(x)
    speed, accel, turn = run_sdd.features_2d(x, y, dt)
    train_len = n // 2
    period = max(2, min(int(30.0 / dt), max(2, n // 8)))
    s_w = max(2, min(10, max(2, n // 30)))
    p_w, half_d_w = 3, 1
    ss_idx = s_w * 2 + half_d_w + period * (p_w - 1)

    motion = np.zeros(n)
    motion_ok = 0
    if train_len - ss_idx >= 15:
        for arr in (speed, accel, turn):
            sc, _, _, _ = gp.fluxev_score(arr, train_len, period,
                                          s_w=s_w, p_w=p_w, half_d_w=half_d_w, q=q)
            if np.any(sc > 0):
                motion_ok += 1
            motion[train_len:] = np.maximum(motion[train_len:], sc)

    dwell = stationary_episode(speed, dt, dwell_lo, dwell_hi)
    gather = gather_score_2d(x, y, others, dt, gather_radius, gather_min) if others else np.zeros(n)
    geof = geofence_score_2d(x, y, zone) if zone else np.zeros(n)
    final = np.maximum.reduce([motion, dwell, gather, geof])
    return dict(speed=speed, accel=accel, turn=turn, motion=motion, dwell=dwell,
                gather=gather, geofence=geof, final=final, n=n, dt=dt,
                motion_ok=motion_ok, train_len=train_len)


# --------------------------- TCN 打分工具 ---------------------------

def frame_ade(model, x, y, obs_len, pred_len):
    """整条轨迹逐帧原始 ADE(米)，头部/尾部无窗口处为 NaN(批量计算)。"""
    n = len(x)
    total = obs_len + pred_len
    if n < total:
        return np.full(n, np.nan)
    obs_list, fut_list = [], []
    for s in range(0, n - total + 1):
        ref = np.array([x[s + obs_len - 1], y[s + obs_len - 1]])
        obs_list.append(np.stack([x[s:s + obs_len], y[s:s + obs_len]], -1) - ref)
        fut_list.append(np.stack([x[s + obs_len:s + total], y[s + obs_len:s + total]], -1) - ref)
    ade_all = T.compute_ade(model, np.asarray(obs_list, np.float32),
                            np.asarray(fut_list, np.float32))
    ade = np.full(n, np.nan)
    ade[obs_len:obs_len + len(ade_all)] = ade_all
    return ade


def zclip(ade, mu, sigma, k=3.0):
    """ADE -> [0,1] z-clip 标定(真零地板，等价于 S_dl 置信度门限)。"""
    ade = np.nan_to_num(ade, nan=mu)
    return np.clip((ade - mu) / (k * sigma + 1e-8), 0.0, 1.0)


def _auroc(gt, score):
    if (gt == 1).any() and (gt == 0).any() and np.std(score) > 1e-9:
        from sklearn import metrics
        return float(metrics.roc_auc_score(gt, score))
    return 0.5


# --------------------------- TCN 训练(在 SDD 正常行走上重训) ---------------------------

def train_sdd_tcn(dt=1.0, obs=16, pred=8, hidden=32, layers=4, k=3, dropout=0.1,
                  epochs=15, batch=256, lr=1e-3, seed=1, max_traj=200, stride=2):
    """用 SDD 行走轨迹重训 TCN(同 tcn_branch 架构/目标)，标定 mu/sigma，落盘 checkpoint。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"扫描 SDD 行走轨迹(dt={dt}s, obs={obs}, pred={pred}) ...")
    tracks = collect_sdd_walking(dt, max_traj, seed)
    if not tracks:
        raise RuntimeError("未找到可用 SDD 行走轨迹")
    trajs = [(x, y) for _, x, y, _ in tracks]
    obs_arr, fut_arr = T.windowize(trajs, obs, pred, stride, 10 ** 9)
    print(f"  行走轨迹数={len(tracks)}，样本窗口数={len(obs_arr)}")

    idx = np.random.default_rng(seed).permutation(len(obs_arr))
    n_val = int(len(obs_arr) * 0.1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr_obs, tr_fut = obs_arr[tr_idx], fut_arr[tr_idx]
    val_obs, val_fut = obs_arr[val_idx], fut_arr[val_idx]
    print(f"  训练 {len(tr_idx)} / 验证 {len(val_idx)}")

    from tcn_branch import TCNPredictor
    model = TCNPredictor(in_dim=2, hidden=hidden, layers=layers, k=k,
                         obs_len=obs, pred_len=pred, dropout=dropout)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量={n_param}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_x = torch.tensor(tr_obs, dtype=torch.float32)
    tr_y = torch.tensor(tr_fut, dtype=torch.float32)
    va_x = torch.tensor(val_obs, dtype=torch.float32)
    va_y = torch.tensor(val_fut, dtype=torch.float32)

    best_va, best_ep, best_state = float("inf"), -1, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_x))
        total, cnt = 0.0, 0
        for i in range(0, len(tr_x), batch):
            b = perm[i:i + batch]
            pr = model(tr_x[b])
            loss = torch.nn.functional.mse_loss(pr, tr_y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(b); cnt += len(b)
        va_loss = float(torch.nn.functional.mse_loss(model(va_x), va_y))
        if va_loss < best_va:
            best_va, best_ep = va_loss, ep + 1
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        print(f"  epoch {ep+1:02d}/{epochs}  train={total/cnt:.4f}  val={va_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  最优 epoch={best_ep}/{epochs} (val_loss={best_va:.4f})")

    val_ade = T.compute_ade(model, val_obs, val_fut)
    stats = {"mu": float(val_ade.mean()), "sigma": float(val_ade.std()), "offset": 2.0}
    cfg = dict(in_dim=2, hidden=hidden, layers=layers, k=k,
               obs_len=obs, pred_len=pred, dropout=dropout)
    T.save_checkpoint(SDD_TCN_CKPT, model, stats, cfg)
    print(f"  标定 mu={stats['mu']:.3f}m sigma={stats['sigma']:.3f}m -> {SDD_TCN_CKPT}")
    return model, stats, cfg


def ensure_sdd_tcn(force=False):
    """加载 SDD 重训的 TCN；不存在则训练。返回 (model, stats, cfg)。"""
    if not force and os.path.exists(SDD_TCN_CKPT):
        return T.load_predictor(SDD_TCN_CKPT)
    return train_sdd_tcn()


if __name__ == "__main__":
    m, s, c = ensure_sdd_tcn()
    print("SDD TCN ready:", c, "mu=%.3f sigma=%.3f" % (s["mu"], s["sigma"]))
