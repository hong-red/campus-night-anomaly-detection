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
"""多人聚集规则示意图 —— 与 fig_dwell_two_conditions.png 同版式。"""

import os, sys
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
RADIUS = 50.0        # GATHER_RADIUS：聚集半径 (m)
MIN_PEOPLE = 3       # GATHER_MIN_PEOPLE：深夜 >= 3 人

t = np.arange(0, 200.0, DT)
n = len(t)
rng = np.random.default_rng(7)


def linseg(tt, t0, t1, p0, p1):
    f = np.clip((tt - t0) / (t1 - t0), 0.0, 1.0)
    return p0 + (p1 - p0) * f


# 主人：原点附近微小游走
main_x = rng.normal(0, 0.6, n)
main_y = rng.normal(0, 0.6, n)

# A：从左上快速接近，t≈40 进入 50 m，之后慢速靠拢主人
ax_ = np.where(t < 55, linseg(t, 0, 55, -88, -30), linseg(t, 55, 200, -30, -12))
ay_ = np.where(t < 55, linseg(t, 0, 55, 30, 18), linseg(t, 55, 200, 18, 15))
ax_ = ax_ + rng.normal(0, 1.0, n)
ay_ = ay_ + rng.normal(0, 1.0, n)

# B：从右下直线接近，速度较慢，t≈100 才进入 50 m
bx_ = linseg(t, 0, 200, 100, -12) + rng.normal(0, 1.0, n)
by_ = linseg(t, 0, 200, -35, -20) + rng.normal(0, 1.0, n)

# 各自到主人的距离 + 是否在 50 m 半径内
dA = np.hypot(ax_ - main_x, ay_ - main_y)
dB = np.hypot(bx_ - main_x, by_ - main_y)
inA = dA <= RADIUS
inB = dB <= RADIUS
t_a = t[np.argmax(inA)]
t_b = t[np.argmax(inB)]
score = np.clip((inA.astype(float) + inB.astype(float)) / (MIN_PEOPLE - 1), 0.0, 1.0)


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


fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 55 * MM),
                         gridspec_kw={"width_ratios": [1.25, 1.0]})

# ---------- 左：地图 ----------
ax = axes[0]
ax.add_patch(plt.Circle((0, 0), RADIUS, fc="#E8272C", ec="#E8272C", alpha=0.10, lw=3.0, zorder=1))

# 他人轨迹：圈外灰、圈内红
for (px, py, d) in [(ax_, ay_, dA), (bx_, by_, dB)]:
    m = d <= RADIUS
    ox, oy = seg(px, py, ~m)
    ix, iy = seg(px, py, m)
    ax.plot(ox, oy, lw=1.3, color="#9a9a9a", zorder=4)
    ax.plot(ix, iy, lw=2.3, color="#E8272C", zorder=4)

# 主人轨迹 + 三个小人（最终都进圈内）
ax.plot(main_x, main_y, lw=1.3, color="#252525", zorder=4)
ax.scatter(ax_[0], ay_[0], s=90, color="#00B140", zorder=6)
ax.scatter(bx_[0], by_[0], s=90, color="#00B140", zorder=6)
person(ax, main_x[-1], main_y[-1], s=5.0)
person(ax, ax_[-1], ay_[-1], s=5.0)
person(ax, bx_[-1], by_[-1], s=5.0)

ax.text(0, RADIUS + 6, "gather radius = 50 m", fontsize=16, ha="center", color="#E8272C", fontweight="bold")
ax.text(ax_[0], ay_[0] - 8, "start A", fontsize=13, ha="center", color="#00B140")
ax.text(bx_[0], by_[0] - 8, "start B", fontsize=13, ha="center", color="#00B140")

ax.set_xlim(-130, 130); ax.set_ylim(-62, 62)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Condition 1: within 50 m radius", fontsize=15, loc="left", pad=10)
ax.grid(alpha=0.2)

# ---------- 右：时间 -> 分数 ----------
ax = axes[1]
ax.step(t, score, where="post", color="#E8272C", lw=2.0)
ax.fill_between(t, 0, score, where=score >= 0.5, color="#E8272C", alpha=0.35, step="post")
ax.axvline(t_a, color="#777777", ls=":", lw=1.2)      # 第 2 人
ax.axvline(t_b, color="#777777", ls="--", lw=1.2)     # 第 3 人
ax.axhline(0.5, color="#999999", ls="--", lw=0.9)
ax.text(t_a, 1.12, "2 people", fontsize=13, ha="center", color="0.35")
ax.text(t_b, 1.12, "≥ 3 people", fontsize=13, ha="center", color="0.35")
ax.text(198, 0.54, "alarm ≥ 0.5", fontsize=12, ha="right", va="bottom", color="0.35")
ax.set_xlabel("time (s)"); ax.set_ylabel("gather score")
ax.set_ylim(-0.02, 1.26); ax.set_xlim(0, 200)
ax.set_title("Condition 2: ≥ 3 people co-present", fontsize=15, loc="left", pad=10)
ax.grid(alpha=0.2)

fig.suptitle("Gathering: come within 50 m AND ≥ 3 people", fontsize=18, y=0.98, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_gather_two_conditions.png", outdir=os.path.join(HERE, "figs"))
print("done:", round(t_a, 1), round(t_b, 1))
