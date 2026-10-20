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
"""图2：只用 FluxEV 跑后三个运动行为 —— 识别率 ≈ 0.5（盲）。

左：三种运动轨迹形态（徘徊 / 疾跑 / 之字形），行人沿轨迹移动。
右：FluxEV 对这三类的 acc@0.5（看着 0.63）vs AUROC（0.50 = 随机）。
"""

import os, sys
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from style import apply_cn_style, C, DOUBLE_COL, MM, save

apply_cn_style()
HERE = os.path.dirname(os.path.abspath(__file__))
INK, RED = "#252525", "#E8272C"

def person(ax, x, y, s=1.0, c=INK):
    ax.add_patch(plt.Circle((x, y + 0.85*s), 0.20*s, fc=c, ec="none", zorder=6))
    ax.plot([x, x], [y + 0.62*s, y + 0.25*s], color=c, lw=2.2*s, zorder=6)
    ax.plot([x - 0.35*s, x + 0.35*s], [y + 0.45*s, y + 0.45*s], color=c, lw=1.5*s, zorder=6)
    ax.plot([x, x - 0.22*s], [y + 0.25*s, y], color=c, lw=1.5*s, zorder=6)
    ax.plot([x, x + 0.22*s], [y + 0.25*s, y], color=c, lw=1.5*s, zorder=6)

def clean(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

fig = plt.figure(figsize=(DOUBLE_COL, 58*MM))

# ---- 左：三种轨迹 + 行人 ----
scenes = [
    ("Wandering", "#0B6BFF", (-0.2, 6.2, -2.0, 2.0)),
    ("Sprint",    "#E8272C", (-0.2, 6.2, -2.0, 2.0)),
    ("Zigzag",    "#FF7A00", (-0.2, 6.2, -2.0, 2.0)),
]
t = np.linspace(0, 6, 300)
for i, (name, color, lim) in enumerate(scenes):
    ax = fig.add_axes([0.04, 0.70 - 0.32*i, 0.42, 0.24])
    if name == "Wandering":
        y = np.sin(t * 2.2) * 1.4
        person(ax, 5.2, y[-1] + 0.6, s=0.55, c=color)
    elif name == "Sprint":
        y = np.zeros_like(t)
        person(ax, 5.2, 0.6, s=0.55, c=color)
    else:
        y = np.where((t * 2 % 2) < 1, 1.2, -1.2)
        person(ax, 5.2, y[-1] + 0.6, s=0.55, c=color)
    ax.plot(t, y, lw=2.0, color=color)
    ax.text(0.0, 1.7, name, fontsize=10, color=color, fontweight="bold", va="top")
    clean(ax, lim[:2], lim[2:])

# ---- 右：FluxEV 结果 ----
ax = fig.add_axes([0.56, 0.12, 0.42, 0.80])
beh = ["Wander", "Sprint", "Zigzag"]
acc = [0.6293, 0.6322, 0.6236]
auroc = [0.4954, 0.5105, 0.4911]
x = np.arange(3); w = 0.34
ax.bar(x - w/2, acc, w, color=C["calm"], label="accuracy (acc@0.5)", zorder=3)
ax.bar(x + w/2, auroc, w, color="#9AA0A6", label="AUROC (honest)", zorder=3)
for i in range(3):
    ax.text(i - w/2, acc[i] + 0.02, f"{acc[i]:.2f}", ha="center", fontsize=8.5)
    ax.text(i + w/2, auroc[i] + 0.02, f"{auroc[i]:.2f}", ha="center", fontsize=8.5)
ax.axhline(0.5, color=RED, ls="--", lw=1.2, zorder=2)
ax.text(2.42, 0.52, "chance 0.5", fontsize=8, color=RED, ha="right", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(beh, fontsize=9)
ax.set_ylabel("score")
ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_axisbelow(True); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("FluxEV (EVT motion) only", fontsize=10, loc="left")

fig.suptitle("FluxEV cannot detect motion behaviours (AUROC ≈ 0.5 = chance)",
             fontsize=12, y=0.98)
save(fig, "fig2_fluxev_motion.png", outdir=os.path.join(HERE, "figs"))
print("done")
