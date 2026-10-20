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
"""对「停留时长分布」做 EVT —— 用 SPOT(GPD) 从正常轨迹的停留时长学出极端阈值，
替代硬编码 dwell 的 120/300s 阈值，并对比两者对原地逗留的 AUROC。

数据口径：
  - 学阈值(GPD 拟合)：所有正常行走轨迹(≥300s)的「停留片段时长」(speed<0.4 的连续时长)，
    未注入任何异常，只描述"正常人会停多久"。
  - 注入测试：在 ≥900s 的长轨迹上，把 55% 处之后的 360s 冻结成原地逗留，
    用 hard(120/300s 硬阈值) 与 EVT时长(学出的 init/extreme 阈值) 各自打分，比 AUROC。
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
from spot_pipe import SPOT

from style import apply_cn_style, C, DOUBLE_COL, MM, panel_label, save
apply_cn_style()

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0
DWELL_SEC = 360
Q = 0.005
HERE = os.path.dirname(os.path.abspath(__file__))


def stay_durations(speed, dt):
    """提取停留片段时长序列（秒）。"""
    durs, c = [], 0.0
    for v in speed:
        if v < gp.V_STATIONARY:
            c += dt
        else:
            if c > 0:
                durs.append(c)
            c = 0.0
    if c > 0:
        durs.append(c)
    return np.asarray(durs)


def stay_run_series(speed, dt):
    """逐帧的"当前已持续停留时长"（秒，因果）。"""
    n = len(speed)
    run = np.zeros(n)
    c = 0.0
    for i in range(n):
        if speed[i] < gp.V_STATIONARY:
            c += dt
        else:
            c = 0.0
        run[i] = c
    return run


def auroc(g, s):
    if g.sum() == 0 or g.sum() == len(g) or np.std(s) < 1e-9:
        return None
    return float(metrics.roc_auc_score(g, s))


def main():
    print("扫描行走轨迹 (min_dur=300s, max_files=2000) ...")
    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=300)
    print(f"共 {len(walks)} 条正常行走轨迹")

    # ---- 1) 从所有正常轨迹提取停留时长，SPOT 学极端阈值 ----
    all_durs = []
    for (npts, med, path, lat5, lon5, tg5) in walks:
        lat, lon, tg = gp.resample(lat5, lon5, tg5, DT)
        speed, _, _ = gp.trajectory_features(lat, lon, DT)
        all_durs.append(stay_durations(speed, DT))
    durs = np.concatenate(all_durs)
    print(f"正常停留片段数 = {len(durs)}  分位数[50/90/95/98/99]% = "
          f"{np.round(np.percentile(durs, [50, 90, 95, 98, 99]), 1)} s")

    spot = SPOT(Q, estimator="MOM")
    spot.fit(durs)
    try:
        spot.initialize(level=0.98, verbose=False)
        t_init = float(spot.init_threshold)
        t_ext = float(spot.extreme_quantile)
        if not np.isfinite(t_ext) or t_ext <= t_init:
            raise ValueError("degenerate GPD")
    except Exception as e:
        print(f"[!] GPD 拟合异常({e})，退回分位数阈值")
        t_init = float(np.percentile(durs, 98))
        t_ext = float(np.percentile(durs, 99.5))

    print(f"\n>>> SPOT(GPD) 学出的停留时长阈值:")
    print(f"    init_threshold(起始抬升) = {t_init:.1f}s")
    print(f"    extreme_quantile(判异常) = {t_ext:.1f}s")
    print(f"    硬编码 dwell 阈值          = 抬升 {gp.DWELL_LOW}s / 满分 {gp.DWELL_MIN}s")

    # ---- 2) 在长轨迹上注入 360s 逗留，对比 hard vs EVT时长 ----
    test_walks = [w for w in walks if w[5][-1] >= 900]
    print(f"\n注入测试用长轨迹(≥900s) = {len(test_walks)} 条")

    agg = {"dwell_hard": [], "dwell_evt": []}
    for (npts, med, path, lat5, lon5, tg5) in test_walks:
        lat, lon, tg = gp.resample(lat5, lon5, tg5, DT)
        n = len(lat)
        if n < int(700 / DT):
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + int(DWELL_SEC / DT), n - 2)
        lat = lat.copy(); lon = lon.copy()
        lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
        gt = np.zeros(n); gt[i0:i1] = 1

        speed, _, _ = gp.trajectory_features(lat, lon, DT)
        run = stay_run_series(speed, DT)
        hard = gp.stationary_episode_score(speed, DT, gp.DWELL_LOW, gp.DWELL_MIN)
        evt = np.clip((run - t_init) / (t_ext - t_init), 0.0, 1.0)

        a_h = auroc(gt, hard)
        a_e = auroc(gt, evt)
        if a_h is not None and a_e is not None:
            agg["dwell_hard"].append(a_h)
            agg["dwell_evt"].append(a_e)

    print(f"\n{'检测器':<16}{'AUROC(均值)':<12}{'最小':<8}{'最大':<8}{'条数'}")
    for name in ("dwell_hard", "dwell_evt"):
        v = agg[name]
        print(f"{name:<16}{np.mean(v):<12.3f}{np.min(v):<8.3f}{np.max(v):<8.3f}{len(v)}")

    a = np.array(agg["dwell_hard"]); b = np.array(agg["dwell_evt"])
    n = min(len(a), len(b))
    up = int((b[:n] > a[:n] + 1e-6).sum())
    down = int((b[:n] < a[:n] - 1e-6).sum())
    print(f"\nEVT时长 vs 硬规则：升 {up} 条 / 降 {down} 条 / 持平 {n - up - down} 条  "
          f"(均值 {np.mean(a[:n]):.3f} -> {np.mean(b[:n]):.3f})")

    out = {
        "n_normal_traj": int(len(walks)),
        "n_stay_episodes": int(len(durs)),
        "learned_init_threshold_s": round(t_init, 1),
        "learned_extreme_quantile_s": round(t_ext, 1),
        "hard_threshold_s": [gp.DWELL_LOW, gp.DWELL_MIN],
        "dwell_hard_auroc_mean": round(float(np.mean(agg["dwell_hard"])), 4),
        "dwell_evt_auroc_mean": round(float(np.mean(agg["dwell_evt"])), 4),
        "n_test_traj": int(len(agg["dwell_hard"])),
    }
    json.dump(out, open(os.path.join(HERE, "dwell_duration_evt.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- 图：停留时长分布(学阈值) + 打分曲线对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 46 * MM))
    ax = axes[0]
    lo = max(durs.min() * 0.9, 1.0)
    hi = max(durs.max(), DWELL_SEC * 1.3)
    bins = np.logspace(np.log10(lo), np.log10(hi), 44)
    ax.hist(durs, bins=bins, color=C["calm"], alpha=0.8, edgecolor="white", lw=0.3)
    ax.set_xscale("log")
    ax.axvline(t_init, color=C["grey"], ls="--", lw=1.0, label=f"EVT 起始 {t_init:.0f}s")
    ax.axvline(t_ext, color=C["alert"], ls="--", lw=1.2, label=f"EVT 极端 {t_ext:.0f}s")
    ax.axvline(gp.DWELL_MIN, color=C["green"], ls=":", lw=1.2, label=f"硬阈值 {gp.DWELL_MIN}s")
    ax.axvline(DWELL_SEC, color=C["dark"], lw=1.3, label=f"注入逗留 {DWELL_SEC}s")
    ax.set_xlabel("正常停留时长 (s, log)")
    ax.set_ylabel("片段数")
    ax.legend(frameon=False, fontsize=6.8)
    panel_label(ax, "a")

    ax = axes[1]
    d = np.linspace(0, DWELL_SEC * 1.3, 400)
    hard_c = np.clip((d - gp.DWELL_LOW) / (gp.DWELL_MIN - gp.DWELL_LOW), 0, 1)
    evt_c = np.clip((d - t_init) / (t_ext - t_init), 0, 1)
    ax.plot(d, hard_c, color=C["green"], lw=1.5, label=f"硬规则 ({gp.DWELL_LOW}/{gp.DWELL_MIN}s)")
    ax.plot(d, evt_c, color=C["alert"], lw=1.5, label=f"EVT时长 ({t_init:.0f}/{t_ext:.0f}s)")
    ax.axhline(0.5, color=C["grey"], ls="--", lw=0.6)
    ax.set_xlabel("当前停留时长 (s)")
    ax.set_ylabel("报警分数")
    ax.legend(frameon=False, fontsize=6.8)
    panel_label(ax, "b")

    fig.suptitle("对停留时长分布做 EVT：学出的阈值 vs 硬编码 5 分钟阈值",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    save(fig, "figC_停留时长EVT.png", outdir=os.path.join(HERE, "figs"))
    print("\nsaved dwell_duration_evt.json + figs/figC_停留时长EVT.png")


if __name__ == "__main__":
    main()
