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
"""两校互为验证集（跨数据集泛化）—— TCN 分支 4 运动行为 × 4 train→test 组合 AUROC。

数据源：F:\Stanford_Drone\results\cross_validate_schools.json（window 级 AUROC，dt=1s）。
组合：
  Geo→Geo 清华训练→清华测试(域内基线)      Geo→SDD 清华训练→斯坦福测试(迁移)
  SDD→SDD 斯坦福训练→斯坦福测试(域内基线)   SDD→Geo 斯坦福训练→清华测试(迁移)
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import apply_cn_style, save

apply_cn_style()

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "results", "cross_validate_schools.json"), encoding="utf-8"))

BEHAVIORS = [("wandering", "Wandering\n徘徊"),
             ("loitering", "Loitering\n原地逗留"),
             ("sprint", "Sprint\n疾跑"),
             ("zigzag", "Zigzag\n之字形")]

COMBOS = [("geo_geo", "Geo→Geo (in-domain)"),
          ("geo_sdd", "Geo→SDD (transfer)"),
          ("sdd_sdd", "SDD→SDD (in-domain)"),
          ("sdd_geo", "SDD→Geo (transfer)")]

COLOR = {"geo_geo": "#9ec5f4", "geo_sdd": "#2a78d6",
         "sdd_sdd": "#f6c9a8", "sdd_geo": "#eb6834"}

fig, ax = plt.subplots(figsize=(10.5, 5.0))
x = np.arange(len(BEHAVIORS))
w = 0.19

for j, (ck, cname) in enumerate(COMBOS):
    off = -1.5 * w + j * w
    vals = [M[k][ck] for k, _ in BEHAVIORS]
    ax.bar(x + off, vals, w, color=COLOR[ck], zorder=3, label=cname)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6)

ax.axhline(0.5, color="#c3c2b7", lw=1.0, ls="--", zorder=2)
ax.text(len(BEHAVIORS) - 0.5, 0.505, "chance 0.5", ha="right", va="bottom",
        fontsize=7.5, color="#898781")

# 原地逗留：结构盲区标注（双向都 < 0.5）
ax.annotate("TCN 结构盲区\n(双向 AUROC < 0.5)",
            xy=(1, 0.13), xytext=(1.35, 0.42),
            fontsize=8.5, color="#B2182B",
            arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in BEHAVIORS], fontsize=9)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)

ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8.5,
          handlelength=1.2, columnspacing=1.0)

ax.set_title("Cross-dataset generalization — TCN branch (two schools as mutual test sets)",
             fontsize=12.5, pad=10)

fig.text(0.01, 0.008,
         "Geolife(清华) 与 Stanford Drone(斯坦福) 各自训练同架构 TCN(dt=1s)，再互为验证集。"
         "徘徊/疾跑/之字形跨校迁移 AUROC 0.65~0.87 ≈ 域内；原地逗留双向 < 0.5，证明是 TCN 结构盲区(须由 FluxEV dwell 规则负责)。",
         fontsize=8, color="#6E6E6E")

fig.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, "cross_validate_schools_paper.png", outdir=os.path.join(HERE, "results"))
print("saved cross_validate_schools_paper.png")
