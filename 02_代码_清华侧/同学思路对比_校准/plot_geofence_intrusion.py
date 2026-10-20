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
"""禁区闯入（电子围栏）规则示意图 —— 与 fig_dwell_two_conditions.png 同版式。"""

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
Z0, Z1 = 60.0, 45.0     # 禁区半宽 / 半高 (m)
RAMP = 30.0             # GEOFENCE_RAMP_M：警戒墙厚度 (m)

# 从左到右穿过禁区：-130 -> +130（1 m/s），越线 t=70、深入 30 m t=100、离开 t=190
t = np.arange(0, 260.0, DT)
n = len(t)
rng = np.random.default_rng(5)
x = -130 + 1.0 * t
y = rng.normal(0, 0.4, n)

inside = (np.abs(x) <= Z0) & (np.abs(y) <= Z1)
depth = np.where(inside, np.minimum.reduce([x + Z0, Z0 - x, y + Z1, Z1 - y]), 0.0)
score = np.clip(depth / RAMP, 0.0, 1.0)


def seg(a, b, m):
    a = np.asarray(a, float).copy(); b = np.asarray(b, float).copy()
    a[~m] = np.nan; b[~m] = np.nan
    return a, b


def person(ax, px, py, s=1.0, color="#252525", zorder=7):
    ax.add_patch(plt.Circle((px, py + 0.9 * s), 0.30 * s, fc=color, ec="none", zorder=zorder))
    ax.plot([px, px], [py + 0.58 * s, py + 0.15 * s], color=color, lw=3.0, zorder=zorder)
    ax.plot([px - 0.5 * s, px + 0.5 * s], [py + 0.42 * s, py + 0.42 * s], color=color, lw=2.0, zorder=zorder)
    ax.plot([px, px - 0.32 * s], [py + 0.15 * s, py - 0.15 * s], color=color, lw=2.0, zorder=zorder)
    ax.plot([px, px + 0.32 * s], [py + 0.15 * s, py - 0.15 * s], color=color, lw=2.0, zorder=zorder)


ox, oy = seg(x, y, ~inside)
ix, iy = seg(x, y, inside)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 55 * MM),
                         gridspec_kw={"width_ratios": [1.25, 1.0]})

# ---------- 左：地图 ----------
ax = axes[0]
ax.add_patch(plt.Rectangle((-Z0, -Z1), 2 * Z0, 2 * Z1, fc="#E8272C", ec="none", alpha=0.10, zorder=1))
clip = PathPatch(Path(np.column_stack([[-Z0, Z0, Z0, -Z0, -Z0], [-Z1, -Z1, Z1, Z1, -Z1]])),
                 transform=ax.transData, facecolor="none", edgecolor="none")
ax.add_patch(clip)
for c in np.arange(-Z0, Z0, 4.5):
    for s in (+1, -1):
        ln, = ax.plot([c, c + 3.0], [-Z1, Z1] if s > 0 else [Z1, -Z1],
                      lw=0.55, color="#E8272C", alpha=0.30, zorder=1)
        ln.set_clip_path(clip)
# 警戒墙内侧 30 m（深入 >= 30 m 才算满报警）
ax.add_patch(plt.Rectangle((-Z0 + RAMP, -Z1 + RAMP), 2 * (Z0 - RAMP), 2 * (Z1 - RAMP),
                           fc="none", ec="#E8272C", lw=1.6, ls=(0, (5, 3)), zorder=3))
ax.add_patch(plt.Rectangle((-Z0, -Z1), 2 * Z0, 2 * Z1, fc="none", ec="#E8272C", lw=3.0, zorder=3))

ax.plot(ox, oy, lw=1.3, color="#9a9a9a", zorder=4)
ax.plot(ix, iy, lw=2.3, color="#E8272C", zorder=4)
ax.scatter(x[0], y[0], s=90, color="#00B140", zorder=6)
ax.scatter(x[-1], y[-1], s=90, marker="s", color="#0B6BFF", zorder=6)

# 闯入者（已在禁区深处）
person(ax, 0, 8, s=5.0)

ax.text(0, Z1 + 9, "restricted zone (geofence)", fontsize=16, ha="center", color="#E8272C", fontweight="bold")
# 警戒墙厚度 30 m
ax.annotate("", xy=(-Z0 + RAMP, 33), xytext=(-Z0, 33),
            arrowprops=dict(arrowstyle="<->", color="#E8272C", lw=1.2))
ax.text(-Z0 + RAMP / 2, 37, "30 m", fontsize=13, ha="center", color="#E8272C")
ax.text(x[0], y[0] - 9, "start", fontsize=13, ha="center", color="#00B140")
ax.text(x[-1], y[-1] - 9, "end", fontsize=13, ha="center", color="#0B6BFF")

ax.set_xlim(-140, 140); ax.set_ylim(-58, 62)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Condition 1: enters the restricted zone", fontsize=15, loc="left", pad=10)
ax.grid(alpha=0.2)

# ---------- 右：深度 -> 分数 ----------
ax = axes[1]
d_ax = np.linspace(-5, 40, 400)
s_ax = np.clip(d_ax / RAMP, 0.0, 1.0)
ax.plot(d_ax, s_ax, color="#E8272C", lw=2.0)
ax.axvspan(0, 40, color="#E8272C", alpha=0.08)
ax.fill_between(d_ax, 0, s_ax, where=s_ax >= 0.5, color="#E8272C", alpha=0.35)
ax.axvline(0, color="#777777", ls=":", lw=1.2)      # 围栏边界
ax.axvline(30, color="#777777", ls="--", lw=1.2)    # 30 m 深
ax.axhline(0.5, color="#999999", ls="--", lw=0.9)
ax.text(0, 1.12, "fence", fontsize=13, ha="center", color="0.35")
ax.text(30, 1.12, "30 m deep", fontsize=13, ha="center", color="0.35")
ax.text(38, 0.54, "alarm ≥ 0.5", fontsize=12, ha="right", va="bottom", color="0.35")
ax.set_xlabel("depth inside fence (m)"); ax.set_ylabel("intrusion score")
ax.set_ylim(-0.02, 1.26); ax.set_xlim(-5, 40)
ax.set_title("Condition 2: penetrates ≥ 30 m deep", fontsize=15, loc="left", pad=10)
ax.grid(alpha=0.2)

fig.suptitle("Intrusion: cross the fence AND penetrate ≥ 30 m", fontsize=18, y=0.98, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_intrusion_two_conditions.png", outdir=os.path.join(HERE, "figs"))
print("done")
