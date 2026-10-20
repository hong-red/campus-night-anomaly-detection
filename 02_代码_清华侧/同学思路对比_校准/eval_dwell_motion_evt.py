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
"""原地逗留 + EVT 运动极值（FluxEV motion）—— 数据 + 图

对比三个检测器对原地逗留的 AUROC（在 ≥900s 长轨迹上注入 360s 逗留）：
  dwell_rule        硬编码 5 分钟时间阈值规则
  motion             EVT 运动极值(FluxEV-SPOT 对速度/加速度/方向)
  max_rule_motion    dwell 规则 + EVT 运动极值 取 max

评估口径：测试段(>= n//2)，因为 FluxEV motion 需要前一半做 GPD 训练。
输出：dwell_motion_evt.json + figs/figD_运动极值_对逗留.png
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

from style import apply_cn_style, C, DOUBLE_COL, MM, panel_label, save, chance_line
apply_cn_style()

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0
DWELL_SEC = 360
HERE = os.path.dirname(os.path.abspath(__file__))


def auroc(g, s):
    if g.sum() == 0 or g.sum() == len(g) or np.std(s) < 1e-9:
        return None
    return float(metrics.roc_auc_score(g, s))


def main():
    print("扫描长行走轨迹 (min_dur=900s, max_files=2000) ...")
    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    print(f"长轨迹 = {len(walks)} 条")

    agg = {"dwell_rule": [], "motion": [], "max_rule_motion": []}
    pool_gt = []
    pool = {"dwell_rule": [], "motion": [], "max_rule_motion": []}
    demo = None
    for (npts, med, path, lat5, lon5, tg5) in walks:
        lat, lon, tg = gp.resample(lat5, lon5, tg5, DT)
        n = len(lat)
        if n < int(700 / DT):
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + int(DWELL_SEC / DT), n - 2)
        lat = lat.copy(); lon = lon.copy()
        lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
        gt = np.zeros(n); gt[i0:i1] = 1

        res = gp.run_pipeline(lat, lon, tg, DT, others=None, zone=None,
                              poly=None, q=0.005)
        dwell = res["dwell"]
        motion = res["motion"]
        mx = np.maximum(dwell, motion)

        mask = np.arange(n) >= n // 2
        g = gt[mask]
        if g.sum() == 0 or g.sum() == len(g):
            continue
        pool_gt.append(g)
        for name, s in (("dwell_rule", dwell), ("motion", motion),
                        ("max_rule_motion", mx)):
            pool[name].append(s[mask])
        ok = True
        for name, s in (("dwell_rule", dwell), ("motion", motion),
                        ("max_rule_motion", mx)):
            a = auroc(g, s[mask])
            if a is None:
                ok = False
            else:
                agg[name].append(a)
        if ok and demo is None:
            demo = (tg, res["speed"], dwell, motion, mx, gt, i0, i1, mask)

    print(f"\n{'检测器':<16}{'AUROC(均值)':<12}{'最小':<8}{'最大':<8}{'条数'}")
    for name in ("dwell_rule", "motion", "max_rule_motion"):
        v = agg[name]
        print(f"{name:<16}{np.mean(v):<12.3f}{np.min(v):<8.3f}{np.max(v):<8.3f}{len(v)}")

    out = {k: round(float(np.mean(v)), 4) for k, v in agg.items() if v}
    out["n_test_traj"] = int(len(agg["dwell_rule"]))

    g_all = np.concatenate(pool_gt)
    print(f"\n逐帧 acc@0.5（池化 {len(g_all)} 帧）：")
    for name in ("dwell_rule", "motion", "max_rule_motion"):
        s_all = np.concatenate(pool[name])
        pred = (s_all >= 0.5).astype(int)
        acc = float((pred == g_all).mean())
        tp = int(((pred == 1) & (g_all == 1)).sum())
        fp = int(((pred == 1) & (g_all == 0)).sum())
        fn = int(((pred == 0) & (g_all == 1)).sum())
        print(f"  {name:<16} acc={acc:.4f}  TP={tp} FP={fp} FN={fn}")
        out[name + "_acc_at_0_5"] = round(acc, 4)

    json.dump(out, open(os.path.join(HERE, "dwell_motion_evt.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- 图：左=示例轨迹时序，右=AUROC 柱 ----
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 48 * MM),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    # (a) 示例轨迹：dwell 规则 vs motion 分数（逗留窗口 ±60s）
    ax = axes[0]
    tg, speed, dwell, motion, mx, gt, i0, i1, mask = demo
    lo, hi = max(0, i0 - 60), min(len(tg), i1 + 60)
    tt = tg[lo:hi] - tg[i0]
    ax.plot(tt, dwell[lo:hi], color=C["green"], lw=1.4, label="dwell 规则")
    ax.plot(tt, motion[lo:hi], color=C["calm"], lw=1.4, label="EVT 运动极值")
    ax.fill_between(tt, 0, 1, where=gt[lo:hi] > 0, color=C["alert"], alpha=0.12)
    ax.axvline(0, color=C["grey"], ls="--", lw=0.6)
    ax.axvline(tg[i1] - tg[i0], color=C["grey"], ls="--", lw=0.6)
    ax.set_xlabel("相对逗留起点的时间 (s)")
    ax.set_ylabel("分数")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.text(0.02, 0.96, "红色阴影=原地逗留真值(360s)", transform=ax.transAxes,
            fontsize=7, va="top", color=C["alert"])
    panel_label(ax, "a")

    # (b) AUROC 柱
    ax = axes[1]
    labels = ["dwell 规则", "EVT 运动极值", "max(规则, 运动)"]
    keys = ["dwell_rule", "motion", "max_rule_motion"]
    vals = [float(np.mean(agg[k])) for k in keys]
    cols = [C["green"], C["calm"], C["dark"]]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.55, color=cols, zorder=3)
    chance_line(ax, 0.5, orient="v")
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e6e6e6", lw=0.5)
    panel_label(ax, "b")

    fig.suptitle("原地逗留 + EVT 运动极值：运动分支对静止无信号，反而拉低",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    save(fig, "figD_运动极值_对逗留.png", outdir=os.path.join(HERE, "figs"))
    print("\nsaved dwell_motion_evt.json + figs/figD_运动极值_对逗留.png")


if __name__ == "__main__":
    main()
