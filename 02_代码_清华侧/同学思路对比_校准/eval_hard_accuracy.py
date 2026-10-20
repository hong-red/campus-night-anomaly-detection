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
"""6 行为 × 3 检测器(硬/软/硬加软) 的准确率 —— 现在只算「硬」(if 规则)。

硬 = max(dwell, gather, geofence) 三个 if 规则取 max。
对 6 个行为逐个注入、逐帧打分，输出 acc@0.5 与 AUROC。

行为(英文=图标签)：
  Intrusion 禁区闯入 / Gathering 多人聚集 / Lingering 原地逗留 /
  Wandering 徘徊 / Sprint 疾跑 / Zigzag 之字形折返

指标口径：逐帧、dt=2s、整条轨迹、多轨迹池化。
dwell 规则用 lo=15s(提升后)、hi=300s。

输出：hard_accuracy_6behaviors.json + figs/figE_六行为_硬准确率.png
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.CODE_TSINGHUA)
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics

import geolife_pipeline as gp
from style import apply_nature_style, C, BRANCH_COLOR, DOUBLE_COL, MM, panel_label, save
apply_nature_style()

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0
DWELL_SEC = 360
DWELL_LO = 15
HERE = os.path.dirname(os.path.abspath(__file__))


def local_xy(lat, lon):
    lat0, lon0 = lat[0], lon[0]
    x = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110540.0
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / 110540.0
    lon = lon0 + x / (111320.0 * np.cos(np.radians(lat0)))
    return lat, lon


def max_stationary_run(speed, dt):
    """连续停留的最长时长(秒)。用于排除有自然长停留的轨迹。"""
    run = best = 0.0
    for v in speed:
        if v < gp.V_STATIONARY:
            run += dt
            best = max(best, run)
        else:
            run = 0.0
    return best


# --- 运动注入器(与 tcn_branch 一致，作用在局部米坐标) ---
def inject_loiter(x, y, i0, i1, amp=8.0, cycle=16):
    cx, cy = float(np.mean(x[i0:i1])), float(np.mean(y[i0:i1]))
    dx, dy = x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]
    norm = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / norm, dy / norm
    idx = np.arange(i0, i1)
    ph = np.sin(2 * np.pi * (idx - i0) / cycle)
    x[i0:i1] = cx + ux * amp * ph
    y[i0:i1] = cy + uy * amp * ph


def inject_run(x, y, i0, i1, factor=2.5):
    dx, dy = x - x[i0], y - y[i0]
    x[i0:i1] = x[i0] + dx[i0:i1] * factor
    y[i0:i1] = y[i0] + dy[i0:i1] * factor


def inject_zigzag(x, y, i0, i1, amp=5.0, cycle=20):
    dx, dy = x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]
    norm = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    idx = np.arange(i0, i1)
    osc = amp * np.sin(2 * np.pi * (idx - i0) / cycle)
    x[i0:i1] += px * osc
    y[i0:i1] += py * osc


# --- 行为注入：返回 (lat, lon, others, zone, gt) ---
def inject_dwell(lat, lon, n):
    i0 = int(n * 0.55); i1 = min(i0 + int(DWELL_SEC / DT), n - 2)
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


def inject_motion(lat, lon, n, kind):
    x, y = local_xy(lat, lon)
    i0 = int(n * 0.55); i1 = min(i0 + max(16, int(n * 0.18)), n - 1)
    if kind == "wandering":
        inject_loiter(x, y, i0, i1)
    elif kind == "sprint":
        inject_run(x, y, i0, i1)
    else:
        inject_zigzag(x, y, i0, i1)
    lat2, lon2 = xy_to_latlon(x, y, lat[0], lon[0])
    gt = np.zeros(n); gt[i0:i1] = 1
    return lat2, lon2, None, None, gt


def hard_score(lat, lon, dt, others, zone):
    speed, _, _ = gp.trajectory_features(lat, lon, dt)
    dwell = gp.stationary_episode_score(speed, dt, DWELL_LO, gp.DWELL_MIN)
    gather = gp.gather_score(lat, lon, others, dt) if others else np.zeros(len(lat))
    geof = gp.geofence_score(lat, lon, zone) if zone else np.zeros(len(lat))
    return np.maximum.reduce([dwell, gather, geof])


BEHAVIORS = [
    ("Intrusion", "geofence", "rule"),
    ("Gathering", "gather", "rule"),
    ("Lingering", "dwell", "rule"),
    ("Wandering", "wandering", "motion"),
    ("Sprint", "sprint", "motion"),
    ("Zigzag", "zigzag", "motion"),
]


def main():
    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    print(f"长轨迹 = {len(walks)} 条")

    result = {}
    for label, key, dom in BEHAVIORS:
        g_pool, s_pool = [], []
        for (npts, med, path, lat5, lon5, tg5) in walks:
            lat, lon, tg = gp.resample(lat5, lon5, tg5, DT)
            n = len(lat)
            if n < int(700 / DT):
                continue
            # 排除有自然长停留(>120s)的轨迹，使注入行为成为唯一异常
            speed0, _, _ = gp.trajectory_features(lat, lon, DT)
            if max_stationary_run(speed0, DT) > 120:
                continue
            if key == "geofence":
                lat, lon, others, zone, gt = inject_geofence(lat, lon, n)
            elif key == "gather":
                lat, lon, others, zone, gt = inject_gather(lat, lon, n)
            elif key == "dwell":
                lat, lon, others, zone, gt = inject_dwell(lat, lon, n)
            else:
                lat, lon, others, zone, gt = inject_motion(lat, lon, n, key)
            s = hard_score(lat, lon, DT, others, zone)
            g_pool.append(gt); s_pool.append(s)

        g = np.concatenate(g_pool); s = np.concatenate(s_pool)
        pred = (s >= 0.5).astype(int)
        acc = float((pred == g).mean())
        tp = int(((pred == 1) & (g == 1)).sum())
        fp = int(((pred == 1) & (g == 0)).sum())
        fn = int(((pred == 0) & (g == 1)).sum())
        auroc = (metrics.roc_auc_score(g, s)
                 if g.sum() not in (0, len(g)) and np.std(s) > 1e-9 else None)
        neg_frac = float((g == 0).mean())  # 全判正常时的"假准确率"
        result[key] = dict(label=label, domain=dom, acc=round(acc, 4),
                           auroc=round(auroc, 4) if auroc is not None else None,
                           neg_frac=round(neg_frac, 4), tp=tp, fp=fp, fn=fn)
        print(f"{label:<10} acc@0.5={acc:.4f}  AUROC={auroc}  "
              f"(负例占比 {neg_frac:.3f})  TP={tp} FP={fp} FN={fn}")

    json.dump(result, open(os.path.join(HERE, "hard_accuracy_6behaviors.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- 图：6 行为 × 3 栏(硬/软/硬加软)，现在只填硬 ----
    labels = [b[0] for b in BEHAVIORS]
    hard = [result[b[1]]["acc"] for b in BEHAVIORS]
    neg_frac = [result[b[1]]["neg_frac"] for b in BEHAVIORS]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 58 * MM))
    x = np.arange(len(labels))
    w = 0.26
    # 软/硬加软占位(待 GPU TCN)
    soft = [0.0] * len(labels)
    fused = [0.0] * len(labels)
    ax.bar(x - w, hard, w, color=BRANCH_COLOR["rules"], label="Hard (if-rules)")
    ax.bar(x, soft, w, color=BRANCH_COLOR["dl"], label="Soft (TCN) — pending")
    ax.bar(x + w, fused, w, color=BRANCH_COLOR["fused"], label="Hard+Soft — pending")
    for xi, (v, nf, dom) in enumerate(zip(hard, neg_frac, [b[2] for b in BEHAVIORS])):
        ax.text(xi - w, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
        if dom == "motion":
            ax.plot([xi - w - 0.16, xi - w + 0.16], [nf, nf], color=C["alert"],
                    lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Accuracy (acc@0.5)")
    ax.set_ylim(0, 1.12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    ax.set_title("Tsinghua (Geolife) — Hard branch accuracy across 6 behaviours",
                 loc="left", fontsize=8.5)
    fig.text(0.02, 0.01,
             "Dashed red = 'always-normal' accuracy (class-imbalance floor for motion behaviours, "
             "not real detection). Soft / Hard+Soft to be filled with GPU TCN results.",
             fontsize=6, color=C["grey"])
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "figE_六行为_硬准确率.png", outdir=os.path.join(HERE, "figs"))
    print("\nsaved hard_accuracy_6behaviors.json + figs/figE_六行为_硬准确率.png")


if __name__ == "__main__":
    main()
