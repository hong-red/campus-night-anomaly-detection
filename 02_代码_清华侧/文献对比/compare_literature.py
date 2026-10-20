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
文献对比实验 v7（最终）：规则域逐帧 + 运动域窗口级
================================================

口径（公平对齐）：
  - 规则域(dwell/geofence/gather)：50 条干净行走轨迹、逐帧、第二半段（规则是逐帧方法）
  - 运动域(wandering/sprint/zigzag)：800 条行走轨迹(seed=1000)、【窗口级】——TCN 是窗口方法，
    三个文献也统一转成同样的窗口口径（stride=2、未来段重叠、决策帧取值）

我们（双分支）：
  - 规则域 → 硬规则 lo=15s（本地 final_3cols_local.json hard 值）
  - 运动域 → 本地 TCN 窗口级（full_run.log：徘徊 0.774 / 疾跑 0.808 / 蛇形 0.658）

三篇文献（现算，纯 numpy）：
  SPOT'17 = POT 上下尾；IJGI'22 = 硬规则+大角度转向；Appl.Sci.'19 = 活动面积(扇形位移)
"""

import os
import sys
import json

_PROJ = _PKGP.CODE_TSINGHUA
_T = os.path.join(_PROJ, "tcn_dl_branch")
for p in (_PROJ, _T):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402
from sklearn import metrics  # noqa: E402

import tcn_branch as T  # noqa: E402
import geolife_pipeline as gp  # noqa: E402

OUTDIR = os.path.join(_PROJ, "results", "lit_compare")
os.makedirs(OUTDIR, exist_ok=True)

DT = 2.0
DWELL_LO = 15.0
DWELL_HI = 300.0
OBS, PRED, STRIDE = 16, 8, 2

# 我们 TCN 运动域（徘徊窗口级 / 疾跑蛇形逐帧，与论文表 lit 一致）
OURS_TCN = {"wandering": 0.774, "sprint": 0.729, "zigzag": 0.547}


# ============================ 规则域注入 ============================

def max_stationary_run(speed, dt):
    run = best = 0.0
    for v in speed:
        if v < gp.V_STATIONARY:
            run += dt
            best = max(best, run)
        else:
            run = 0.0
    return best


def inject_dwell(lat, lon, n):
    i0 = int(n * 0.55); i1 = min(i0 + int(360 / DT), n - 2)
    lat = lat.copy(); lon = lon.copy()
    lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
    gt = np.zeros(n); gt[i0:i1] = 1
    return lat, lon, None, None, gt


def inject_geofence(lat, lon, n):
    c = int(n * 0.7)
    zone = (float(lat[c]), float(lon[c]), 60.0)
    gt = (gp.geofence_score(lat, lon, zone) > 0).astype(int)
    return lat, lon, None, zone, gt


def inject_gather(lat, lon, n):
    g0 = int(n * 0.55); g1 = min(g0 + int(300 / DT), n - 1)
    kx = 111320.0 * np.cos(np.radians(lat[g0])); ky = 110540.0
    others = []
    for dx, dy in ((12.0, 0.0), (0.0, 12.0)):
        o_lat = np.full(n, lat[g0] + 200.0 / ky)
        o_lon = np.full(n, lon[g0] + 200.0 / kx)
        o_lat[g0:g1] = lat[g0:g1] + dy / ky
        o_lon[g0:g1] = lon[g0:g1] + dx / kx
        others.append((o_lat, o_lon))
    gt = np.zeros(n); gt[g0:g1] = 1
    return lat, lon, others, None, gt


# ============================ 三篇文献复现 ============================

def spot_vanilla_score(speed, dt, q=0.005, train_frac=0.5):
    n = len(speed)
    m = int(n * train_frac)
    score = np.zeros(n)
    if m < 30:
        return score

    def _pot_tail(signal, q):
        calib = signal[:m].astype(float)
        if calib.size < 30:
            return None
        t0 = float(np.quantile(calib, 0.98))
        exc = calib[calib > t0] - t0
        if exc.size < 3:
            return None
        try:
            c, _, scale = st.genpareto.fit(exc, floc=0)
        except Exception:
            return None
        if not np.isfinite(c) or not np.isfinite(scale) or scale <= 0 or abs(c) < 1e-9:
            return None
        Nt = float(exc.size)
        z = t0 + (scale / c) * (((q * n) / Nt) ** (-c) - 1.0)
        if not np.isfinite(z) or z <= t0:
            return None
        return t0, z

    up = _pot_tail(speed, q)
    lo = _pot_tail(-speed, q)
    for i in range(m, n):
        s = 0.0
        if up is not None and speed[i] > up[0]:
            s = max(s, min(1.0, (speed[i] - up[0]) / (up[1] - up[0])))
        if lo is not None and -speed[i] > lo[0]:
            s = max(s, min(1.0, ((-speed[i]) - lo[0]) / (lo[1] - lo[0])))
        score[i] = s
    return score


def _heading_deg(lat, lon):
    n = len(lat)
    h = np.zeros(n)
    for i in range(1, n - 1):
        h[i] = np.degrees(np.arctan2(lat[i + 1] - lat[i - 1], lon[i + 1] - lon[i - 1]))
    h[0] = h[1] if n > 1 else 0.0
    h[-1] = h[-2] if n > 2 else h[-1]
    return h


def ijgi_angle_wander(lat, lon, dt, speed, angle_thr=150.0, window=120.0):
    n = len(lat)
    dh = np.abs(np.diff(np.unwrap(np.radians(_heading_deg(lat, lon))))) * 180.0 / np.pi
    dh = np.concatenate(([0.0], dh))
    turn = np.zeros(n)
    for i in range(2, n - 2):
        if speed[i] > 0.4 and dh[i - 1:i + 3].sum() > angle_thr:
            turn[i] = 1.0
    w = max(1, int(window / dt))
    score = np.zeros(n)
    run = 0.0
    for i in range(n):
        run += turn[i]
        if i >= w:
            run -= turn[i - w]
        score[i] = min(1.0, run / 3.0)
    return score


def hard_rules_score(lat, lon, dt, others, zone):
    speed, _, _ = gp.trajectory_features(lat, lon, dt)
    dwell = gp.stationary_episode_score(speed, dt, DWELL_LO, DWELL_HI)
    gather = gp.gather_score(lat, lon, others, dt) if others else np.zeros(len(lat))
    geof = gp.geofence_score(lat, lon, zone) if zone else np.zeros(len(lat))
    return np.maximum.reduce([dwell, gather, geof])


def ijgi_score(lat, lon, dt, speed, others, zone):
    return np.maximum(hard_rules_score(lat, lon, dt, others, zone),
                      ijgi_angle_wander(lat, lon, dt, speed))


def applsci_sector_score(x, y, dt, T0=60.0, D_thr=30.0):
    n = len(x)
    w = max(1, int(T0 / dt))
    dmax = np.zeros(n)
    for i in range(n):
        lo = max(0, i - w)
        dmax[i] = np.hypot(x[lo:i + 1] - x[lo], y[lo:i + 1] - y[lo]).max()
    return np.clip((D_thr - dmax) / D_thr, 0.0, 1.0)


def ref_frame_scores(lat, lon, dt, others, zone, x, y):
    speed, _, _ = gp.trajectory_features(lat, lon, dt)
    return {
        "rules": hard_rules_score(lat, lon, dt, others, zone),
        "spot": spot_vanilla_score(speed, dt),
        "ijgi": ijgi_score(lat, lon, dt, speed, others, zone),
        "applsci": applsci_sector_score(x, y, dt),
    }


# ============================ 窗口级 ============================

def window_scores(frame_score, gt):
    """窗口级：未来段 [s+OBS, s+OBS+PRED) 与异常重叠则标签=1，分数=决策帧 s+OBS。"""
    n = len(gt)
    total = OBS + PRED
    ws, wl = [], []
    for s in range(0, n - total + 1, STRIDE):
        fut = np.arange(s + OBS, s + total)
        wl.append(1 if gt[fut].sum() > 0 else 0)
        ws.append(frame_score[s + OBS])
    return np.array(ws), np.array(wl)


def pooled_auroc(ws_list, wl_list):
    ws = np.concatenate(ws_list)
    wl = np.concatenate(wl_list)
    if wl.sum() == 0 or wl.sum() == len(wl) or np.std(ws) < 1e-9:
        return 0.5
    return float(metrics.roc_auc_score(wl, ws))


# ============================ 基准 ============================

def run_rule():
    walks = gp.scan_walking_trajectories(T.DEFAULT_DATA, max_files=2000, min_dur=900)
    clean = []
    for w in walks:
        lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
        speed0, _, _ = gp.trajectory_features(lat, lon, DT)
        if max_stationary_run(speed0, DT) <= 120:
            clean.append(w)
    print(f"  规则域干净轨迹 = {len(clean)} 条")

    events = {"dwell": inject_dwell, "geofence": inject_geofence, "gather": inject_gather}
    result = {}
    for key, inj in events.items():
        pools = {d: ([], []) for d in ["rules", "spot", "ijgi", "applsci"]}
        for w in clean:
            lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
            n = len(lat)
            if n < int(700 / DT):
                continue
            lat, lon, others, zone, gt = inj(lat, lon, n)
            x, y = T.latlon_to_local(lat, lon)
            sc = ref_frame_scores(lat, lon, DT, others, zone, x, y)
            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            for d, s in sc.items():
                pools[d][0].append(g)
                pools[d][1].append(s[mask])
        result[key] = {d: round(float(metrics.roc_auc_score(
            np.concatenate(pools[d][0]), np.concatenate(pools[d][1]))), 3) for d in pools}
    return result


def run_motion_window():
    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, 800, seed=1000)
    print(f"  运动域评估轨迹 = {len(trajs)} 条 (seed=1000)")

    events = {"wandering": T.inject_loiter, "sprint": T.inject_run, "zigzag": T.inject_zigzag}
    DETS = ["ijgi", "applsci", "spot"]
    pool = {(en, d): ([], []) for en in events for d in DETS}
    for en, fn in events.items():
        for t in trajs:
            x, y, lat, lon, tg, _ = t
            n = len(x)
            if n < OBS + PRED + 30:
                continue
            i0 = int(n * 0.55)
            i1 = min(i0 + max(16, int(n * 0.18)), n - OBS - PRED - 2)
            xa, ya = x.copy(), y.copy()
            fn(xa, ya, i0, i1)
            gt = np.zeros(n); gt[i0:i1] = 1
            lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
            sc = ref_frame_scores(lat_a, lon_a, DT, None, None, xa, ya)
            for d in DETS:
                if en == "wandering":
                    # 徘徊：窗口级（TCN 自然口径，对齐论文表6）
                    ws, wl = window_scores(sc[d], gt)
                    pool[(en, d)][0].append(ws)
                    pool[(en, d)][1].append(wl)
                else:
                    # 疾跑/蛇形：逐帧（第二半段）
                    mask = np.arange(n) >= n // 2
                    g = gt[mask]
                    if g.sum() == 0 or g.sum() == len(g):
                        continue
                    pool[(en, d)][0].append(sc[d][mask])   # 分数
                    pool[(en, d)][1].append(g)             # 标签
    return {en: {d: round(pooled_auroc(pool[(en, d)][0], pool[(en, d)][1]), 3)
                 for d in DETS} for en in events}


# ============================ 组装 ============================

def main():
    loc = json.load(open(os.path.join(_PROJ, "同学思路对比_校准", "final_3cols_local.json"),
                         encoding="utf-8"))
    OURS_RULE = {
        "dwell": loc["dwell"]["hard"]["auroc"],
        "geofence": loc["geofence"]["hard"]["auroc"],
        "gather": loc["gather"]["hard"]["auroc"],
    }

    print("[1/2] 规则域（50 条，逐帧）...")
    rule = run_rule()

    print("[2/2] 运动域（800 条，窗口级，三篇文献）...")
    motion = run_motion_window()

    order = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag"]
    matrix = {}
    for e in order:
        if e in rule:
            matrix[e] = {"ours_dual": round(OURS_RULE[e], 3), "ours_branch": "rules(lo=15)",
                         "ijgi": rule[e]["ijgi"], "applsci": rule[e]["applsci"],
                         "spot": rule[e]["spot"]}
        else:
            matrix[e] = {"ours_dual": OURS_TCN[e],
                         "ours_branch": "TCN(window)" if e == "wandering" else "TCN(frame)",
                         "ijgi": motion[e]["ijgi"], "applsci": motion[e]["applsci"],
                         "spot": motion[e]["spot"]}

    methods = ["ours_dual", "ijgi", "applsci", "spot"]
    summary = {}
    for m in methods:
        vals = [matrix[e][m] for e in order]
        cover = sum(1 for v in vals if v > 0.55)
        summary[m] = dict(coverage=f"{cover}/6", n_covered=cover)

    json.dump({"order": order, "matrix": matrix, "summary": summary},
              open(os.path.join(OUTDIR, "literature_comparison.json"), "w"),
              ensure_ascii=False, indent=2)

    M_LABEL = {"ours_dual": "Ours (dual-branch)", "ijgi": "IJGI'22 (rules)",
               "applsci": "Appl.Sci.'19 (area)", "spot": "SPOT'17 (EVT)"}
    print("\n" + "=" * 84)
    print("AUROC 矩阵（6 事件 × 4 方法；规则域逐帧 / 运动域窗口级）")
    print("=" * 84)
    print(f"{'event':<12}" + "".join(f"{M_LABEL[m]:<22}" for m in methods))
    print("-" * 84)
    for e in order:
        print(f"{e:<12}" + "".join(f"{matrix[e][m]:<22.3f}" for m in methods)
              + f"  [ours={matrix[e]['ours_branch']}]")
    print("-" * 84)
    for m in methods:
        print(f"  {M_LABEL[m]:<22} coverage={summary[m]['coverage']}")
    print("\n已保存:", os.path.join(OUTDIR, "literature_comparison.json"))


if __name__ == "__main__":
    main()
