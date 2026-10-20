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
"""文献对比实验 —— 出版级图表（读取 literature_comparison.json，6 事件版）。

图1：6 事件 × 4 方法 AUROC 分组条形图（随机 0.5 虚线；覆盖率并入图例）
图2：汇总 —— (b) 覆盖事件数  (c) 部署属性对照表
"""

import os
import sys
import json

_PROJ = _PKGP.CODE_TSINGHUA
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib.pyplot as plt
from style import apply_nature_style, C, DOUBLE_COL, MM, panel_label, save, chance_line

apply_nature_style()

OUTDIR = os.path.join(_PROJ, "results", "lit_compare")
data = json.load(open(os.path.join(OUTDIR, "literature_comparison.json"), encoding="utf-8"))
order = data["order"]
matrix = data["matrix"]
summary = data["summary"]

METHODS = ["ours_dual", "ijgi", "applsci", "spot"]
M_BASE = {"ours_dual": "Ours (dual-branch)", "ijgi": "IJGI'22 (rule-based)",
          "applsci": "Appl.Sci.'19 (area)", "spot": "SPOT'17 (EVT)"}
M_COLOR = {"ours_dual": "#0b0b0b", "ijgi": "#1B7837", "applsci": "#eb6834", "spot": "#4393C3"}
E_LABEL = {"dwell": "Dwell\n(loiter)", "geofence": "Geofence\n(intrusion)",
           "gather": "Gather\n(>=3 ppl)", "wandering": "Wandering\n(window)",
           "sprint": "Sprint", "zigzag": "Zigzag"}


def _label(m):
    return f"{M_BASE[m]}  [{summary[m]['coverage']}]"


# ============================ 图1：AUROC 矩阵 ============================
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 62 * MM))
x = np.arange(len(order))
n_m = len(METHODS)
w = 0.80 / n_m
for j, m in enumerate(METHODS):
    off = (j - (n_m - 1) / 2) * w
    vals = [matrix[e][m] for e in order]
    ax.bar(x + off, vals, w, color=M_COLOR[m], zorder=3,
           edgecolor="white", linewidth=0.4, label=_label(m))
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.014, f"{v:.3f}", ha="center", va="bottom",
                fontsize=4.8, color="#333333")
chance_line(ax, 0.5)
ax.set_xticks(x)
ax.set_xticklabels([E_LABEL[e] for e in order], fontsize=6.8)
ax.set_ylabel("AUROC")
ax.set_ylim(0, 1.16)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.5)
ax.legend(frameon=False, loc="upper right", ncol=1, fontsize=6.0,
          handlelength=1.2, handleheight=0.7)
ax.set_title("Literature comparison — AUROC across 6 real-trajectory events",
             loc="left", fontsize=8.5)
panel_label(ax, "a")
fig.text(0.02, 0.02,
         "All events scored frame-wise except wandering (window-level, the TCN's natural granularity). "
         "Dashed line = chance 0.5.",
         fontsize=5.8, color=C["grey"])
fig.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, "fig1_auroc_matrix.png", outdir=OUTDIR)

# ============================ 图2：汇总面板 ============================
fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, 74 * MM),
                         gridspec_kw={"height_ratios": [0.9, 1.15]})

# (b) 覆盖率
ax = axes[0]
covs = [summary[m]["n_covered"] for m in METHODS]
ax.bar(range(len(METHODS)), covs, 0.5, color=[M_COLOR[m] for m in METHODS], zorder=3)
for xi, c in enumerate(covs):
    ax.text(xi, c + 0.1, f"{c}/6", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.axhline(6, color="#e0e0e0", lw=0.6, zorder=2)
ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels([M_BASE[m] for m in METHODS], fontsize=6.5)
ax.set_ylabel("Events covered")
ax.set_ylim(0, 6.9)
ax.set_yticks([0, 2, 4, 6])
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.5)
ax.set_title("Coverage (AUROC > 0.55) across 6 events", loc="left", fontsize=8.5)
panel_label(ax, "b")

# (c) 部署属性表 —— 手动定位，避免 ax.table 文字溢出遮挡
ax = axes[1]
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)

header = ["Method", "Camera", "Train", "Labels", "Covers", "Edge"]
rows = [
    ["Ours (dual)", "No", "CPU", "No", "6/6", "Yes"],
    ["SPOT'17", "No", "None", "No", "2/6", "Yes"],
    ["IJGI'22", "No", "None", "Hist.", "4/6", "Yes"],
    ["Appl.Sci.", "Yes", "None", "No", "3/6", "No"],
]
cx = [1.55, 3.35, 4.85, 6.15, 7.75, 9.1]   # 各列中心 x

# 表头
for j, h in enumerate(header):
    ax.text(cx[j], 4.15, h, ha="center", va="center", fontsize=6.5,
            fontweight="bold", color="white", zorder=3)
ax.add_patch(plt.Rectangle((0.35, 3.72), 9.3, 0.86, facecolor="#333333",
                           edgecolor="none", zorder=1))

# 数据行
for i, row in enumerate(rows):
    y = 3.02 - i * 0.82
    for j, v in enumerate(row):
        ax.text(cx[j], y, v, ha="center", va="center", fontsize=6.5,
                fontweight="bold" if j == 0 else "normal", zorder=3)
    if i == 0:
        ax.add_patch(plt.Rectangle((0.35, y - 0.40), 9.3, 0.80,
                                   facecolor="#EAF2E3", edgecolor="none", zorder=1))
    ax.plot([0.35, 9.65], [y - 0.40, y - 0.40], color="#dcdcdc", lw=0.4, zorder=2)

ax.set_title("Deployment & scope", loc="left", fontsize=8.5)
panel_label(ax, "c")
fig.tight_layout()
save(fig, "fig2_summary.png", outdir=OUTDIR)

print("已保存 fig1_auroc_matrix.png / fig2_summary.png 到", OUTDIR)
