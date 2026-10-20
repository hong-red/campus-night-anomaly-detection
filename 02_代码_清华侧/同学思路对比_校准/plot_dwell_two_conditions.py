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
"""原地逗留规则示意图（重做：短轨迹 + 框内小人 + 标签全放框外）。"""

import os, sys
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from style import apply_cn_style, DOUBLE_COL, MM, save

apply_cn_style()
# 字体加粗加大：改用微软雅黑(真粗体)并整体放大 1.5x
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
for _k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
    plt.rcParams[_k] *= 1.5
HERE = os.path.dirname(os.path.abspath(__file__))

DT = 1.0
V_STAT, LO, HI = 0.4, 120.0, 300.0
BOX = 3.0

t = np.arange(0, 480.0, DT)
n = len(t)
rng = np.random.default_rng(3)
x = np.where(t < 60, -12 + 0.15 * t, np.where(t < 420, 0.0, 0.15 * (t - 420)))
y = rng.normal(0, 0.18, n)
x[60:420] = rng.normal(0, 0.13, 360)
y[60:420] = rng.normal(0, 0.13, 360)
speed = np.where((t >= 60) & (t < 420), 0.0, 0.15)
run = np.zeros(n); c = 0.0
for i in range(n):
    c = c + DT if speed[i] < V_STAT else 0.0
    run[i] = c
dwell = np.clip((run - LO) / (HI - LO), 0.0, 1.0)
inside = (np.abs(x) <= BOX) & (np.abs(y) <= BOX)

def seg(a, b, m):
    a = np.asarray(a, float).copy(); b = np.asarray(b, float).copy()
    a[~m] = np.nan; b[~m] = np.nan
    return a, b
ox, oy = seg(x, y, ~inside); ix, iy = seg(x, y, inside)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 55 * MM),
                         gridspec_kw={"width_ratios": [1.25, 1.0]})

# ---------- 左：地图 ----------
ax = axes[0]
ax.add_patch(plt.Rectangle((-BOX, -BOX), 2*BOX, 2*BOX, fc="#E8272C", ec="none", alpha=0.10, zorder=1))
clip = PathPatch(Path(np.column_stack([[-BOX, BOX, BOX, -BOX, -BOX], [-BOX, -BOX, BOX, BOX, -BOX]])),
                 transform=ax.transData, facecolor="none", edgecolor="none")
ax.add_patch(clip)
for c in np.arange(-BOX, BOX, 0.6):
    for s in (+1, -1):
        ln, = ax.plot([c, c + 0.5], [-BOX, BOX] if s > 0 else [BOX, -BOX],
                      lw=0.55, color="#E8272C", alpha=0.30, zorder=1)
        ln.set_clip_path(clip)
ax.add_patch(plt.Rectangle((-BOX, -BOX), 2*BOX, 2*BOX, fc="none", ec="#E8272C", lw=3.0, zorder=3))

ax.plot(ox, oy, lw=1.3, color="#9a9a9a", zorder=4)
ax.plot(ix, iy, lw=2.3, color="#E8272C", zorder=4)
ax.scatter(x[0], y[0], s=90, color="#00B140", zorder=6)
ax.scatter(x[-1], y[-1], s=90, marker="s", color="#0B6BFF", zorder=6)

# 框内小人
ax.add_patch(plt.Circle((0, 0.9), 0.30, fc="#252525", ec="none", zorder=7))
ax.plot([0, 0], [0.58, 0.15], color="#252525", lw=3.0, zorder=7)
ax.plot([-0.5, 0.5], [0.42, 0.42], color="#252525", lw=2.0, zorder=7)
ax.plot([0, -0.32], [0.15, -0.15], color="#252525", lw=2.0, zorder=7)
ax.plot([0, 0.32], [0.15, -0.15], color="#252525", lw=2.0, zorder=7)

ax.text(0, 4.6, "6 m box", fontsize=16, ha="center", color="#E8272C", fontweight="bold")
ax.text(-12, -2.4, "start", fontsize=13, ha="center", color="#00B140")
ax.text(12, -2.4, "end", fontsize=13, ha="center", color="#0B6BFF")

ax.set_xlim(-15, 15); ax.set_ylim(-4.2, 5.6)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Condition 1: stays inside a box", fontsize=15, loc="left", pad=10)
ax.grid(alpha=0.2)

# ---------- 右：时长 → 分数 ----------
ax = axes[1]
ax.plot(t, dwell, color="#E8272C", lw=2.0)
ax.axvspan(60, 420, color="#E8272C", alpha=0.08)
ax.fill_between(t, 0, dwell, where=dwell >= 0.5, color="#E8272C", alpha=0.35)
ax.axvline(60 + LO, color="#777777", ls=":", lw=1.2)
ax.axvline(60 + HI, color="#777777", ls="--", lw=1.2)
ax.axhline(0.5, color="#999999", ls="--", lw=0.9)
ax.text(60 + LO, 1.12, "120 s", fontsize=13, ha="center", color="0.35")
ax.text(60 + HI, 1.12, "300 s\n(5 min)", fontsize=13, ha="center", color="0.35")
ax.text(478, 0.54, "alarm ≥ 0.5", fontsize=12, ha="right", va="bottom", color="0.35")
ax.set_xlabel("time (s)"); ax.set_ylabel("dwell score")
ax.set_ylim(-0.02, 1.26); ax.set_xlim(0, 480)
ax.set_title("Condition 2: stays long enough (5 min)", fontsize=15, loc="left", pad=10)
ax.grid(alpha=0.2)

fig.suptitle("Lingering: stay in a box AND stay 5 min", fontsize=18, y=0.98, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_dwell_two_conditions.png", outdir=os.path.join(HERE, "figs"))
print("done")
