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
"""文献对比图（7 事件 × 4 方法）—— 出版级重绘版（加粗加厚）。
无覆盖率、无底部小字、无 chance 0.5 参考线与标注；柱顶数值两位小数。
数据源：07_对比实验/结果_JSON/literature_comparison_7events.json
输出：07_对比实验/图/fig_lit_compare_7events.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "结果_JSON", "literature_comparison_7events.json")
OUTDIR = os.environ.get("LIT7_OUTDIR", os.path.join(HERE, "图"))
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.weight": "heavy",
    "axes.labelweight": "heavy",
    "axes.titleweight": "heavy",
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size": 4.2,
    "ytick.major.size": 4.2,
})

d = json.load(open(JSON, encoding="utf-8"))
order = d["order"]
labels = d["labels"]
matrix = d["matrix"]

METHODS = ["ours", "spot", "gradings", "ae"]
M_LABEL = {"ours": "Ours (dual-branch)", "spot": "SPOT'17 (EVT)",
           "gradings": "GRADINGS'20 (NF)", "ae": "ICHCI'21 (AE)"}
M_COLOR = {"ours": "#0b0b0b", "spot": "#4393C3", "gradings": "#eb6834", "ae": "#1B7837"}

fig, ax = plt.subplots(figsize=(9.2, 4.3))
x = np.arange(len(order))
n = len(METHODS)
w = 0.80 / n
for j, m in enumerate(METHODS):
    off = (j - (n - 1) / 2.0) * w
    vals = [matrix[e][m] for e in order]
    ax.bar(x + off, vals, w, color=M_COLOR[m], zorder=3,
           edgecolor="white", linewidth=0.6, label=M_LABEL[m])
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                fontsize=6.4, color="#1a1a1a", fontweight="heavy")

ax.set_xticks(x)
ax.set_xticklabels([labels[e] for e in order], fontsize=8.8, fontweight="heavy")
ax.set_ylabel("AUROC", fontsize=11.5, fontweight="heavy")
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="y", labelsize=9.2)
for t in ax.get_yticklabels():
    t.set_fontweight("heavy")
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.8)
ax.legend(frameon=False, loc="upper right", ncol=1, fontsize=8.6,
          handlelength=1.3, handleheight=0.8, labelspacing=0.35)
for t in ax.get_legend().get_texts():
    t.set_fontweight("heavy")
ax.set_title("Literature comparison \u2014 AUROC across 7 real-trajectory events",
             loc="left", fontsize=11.0, fontweight="heavy")

fig.tight_layout()
p = os.path.join(OUTDIR, "fig_lit_compare_7events.png")
fig.savefig(p, dpi=300, bbox_inches="tight")
print("saved", p)