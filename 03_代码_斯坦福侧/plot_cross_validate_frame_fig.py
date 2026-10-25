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
"""跨校验证（帧级）4 行为 x 4 训练->测试组合 AUROC 柱状图 —— 加厚加粗版。

数据源：06_消融实验/结果_JSON/cross_validate_schools_frame.json
  （由 03_代码_斯坦福侧/cross_validate_schools_frame.py 生成，逐帧 ADE 打分、逐帧 AUROC）
组合：
  Geo->Geo 清华训练->清华测试(域内基线)    Geo->SDD 清华训练->斯坦福测试(迁移)
  SDD->SDD 斯坦福训练->斯坦福测试(域内)    SDD->Geo 斯坦福训练->清华测试(迁移)
输出：08_实验结论/最终图/cross_validate_schools_frame.png
"""
import os
import sys
import json

sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import save

# 中文 + 全量加厚加粗
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams.update({
    "font.weight": "heavy",
    "axes.labelweight": "heavy",
    "axes.titleweight": "heavy",
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size": 4.2,
    "ytick.major.size": 4.2,
    "axes.unicode_minus": False,
})

_JSON = os.path.join(_PKGP.ROOT, "06_消融实验", "结果_JSON",
                     "cross_validate_schools_frame.json")
M = json.load(open(_JSON, encoding="utf-8"))

BEHAVIORS = [("wandering", "Wandering"),
             ("loitering", "Loitering"),
             ("sprint", "Sprint"),
             ("zigzag", "Zigzag")]

COMBOS = [("geo_geo", "Geo→Geo (in-domain)"),
          ("geo_sdd", "Geo→SDD (transfer)"),
          ("sdd_sdd", "SDD→SDD (in-domain)"),
          ("sdd_geo", "SDD→Geo (transfer)")]

COLOR = {"geo_geo": "#9ec5f4", "geo_sdd": "#2a78d6",
         "sdd_sdd": "#f6c9a8", "sdd_geo": "#eb6834"}

fig, ax = plt.subplots(figsize=(11.5, 5.6))
x = np.arange(len(BEHAVIORS))
slot = 0.20
w = 0.175

for j, (ck, cname) in enumerate(COMBOS):
    off = (j - 1.5) * slot
    vals = [M[k][ck] for k, _ in BEHAVIORS]
    ax.bar(x + off, vals, w, color=COLOR[ck], zorder=3,
           edgecolor="white", linewidth=0.8, label=cname)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7.6, color="#1a1a1a", fontweight="heavy")

# Loitering blind spot: placed in the empty band above the Loitering group,
# centred on x=1 so it sits in the gap between neighbouring groups (no bar overlap)
ax.annotate("TCN blind spot (< 0.5)",
            xy=(1.0, 0.50), xytext=(1.0, 0.585),
            fontsize=9.5, color="#B2182B", fontweight="heavy",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1.4,
                            shrinkA=1.0, shrinkB=2.0))

ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in BEHAVIORS], fontsize=11, fontweight="heavy")
ax.set_ylabel("AUROC", fontsize=13, fontweight="heavy")
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="y", labelsize=10.5)
for t in ax.get_yticklabels():
    t.set_fontweight("heavy")
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.8)

ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005),
          ncol=4, fontsize=10.0, handlelength=1.3, handleheight=0.9,
          columnspacing=1.6)
for t in ax.get_legend().get_texts():
    t.set_fontweight("heavy")

ax.set_title("Frame-level cross-dataset generalization (AUROC)",
             loc="center", fontsize=14, fontweight="heavy", pad=34)

fig.tight_layout(rect=[0, 0.0, 1, 1])

outdirs = [os.path.join(_PKGP.ROOT, "08_实验结论", "最终图"),
           os.path.join(_PKGP.ROOT, "06_消融实验", "图")]
for od in outdirs:
    save(fig, "cross_validate_schools_frame.png", outdir=od)
print("saved cross_validate_schools_frame.png")
