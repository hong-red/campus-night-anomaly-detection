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
"""最终图：六行为 × 三栏(硬/软/硬加软)，纵轴准确率。

硬 = if 规则（hard_accuracy_6behaviors.json）
软 = TCN（tcn_gpu.json，GPU 结果）
硬加软 = max(硬,软)（fusion_6behaviors.json，本地 TCN 逐帧分数，近似）
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import apply_nature_style, C, BRANCH_COLOR, DOUBLE_COL, MM, save
apply_nature_style()

HERE = os.path.dirname(os.path.abspath(__file__))
hard = json.load(open(os.path.join(HERE, "hard_accuracy_6behaviors.json"), encoding="utf-8"))
soft = json.load(open(os.path.join(HERE, "tcn_gpu.json"), encoding="utf-8"))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering", "dwell": "Lingering",
         "wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag"}

labels = [LABEL[k] for k in ORDER]
hard_v = [hard[k]["acc"] for k in ORDER]
soft_v = [soft[k]["acc"] for k in ORDER]
# 硬加软 = max(硬准确率, 软准确率)
fused_v = [max(hard[k]["acc"], soft[k]["acc"]) for k in ORDER]

fig, ax = plt.subplots(figsize=(DOUBLE_COL, 62 * MM))
x = np.arange(len(labels))
w = 0.24

ax.bar(x - w, hard_v, w, color=BRANCH_COLOR["rules"], label="Hard (if-rules)", zorder=3)
ax.bar(x, soft_v, w, color=BRANCH_COLOR["dl"], label="Soft (TCN)", zorder=3)
ax.bar(x + w, fused_v, w, color=BRANCH_COLOR["fused"], label="Hard+Soft (max)", zorder=3)

for xi, v in enumerate(hard_v):
    ax.text(xi - w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0)
for xi, v in enumerate(soft_v):
    ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0)
for xi, v in enumerate(fused_v):
    ax.text(xi + w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Accuracy (acc@0.5)")
ax.set_ylim(0, 1.18)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.5)

ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.03),
          ncol=3, fontsize=6.3, handlelength=1.1, columnspacing=1.1)

ax.set_title("Tsinghua (Geolife) — accuracy across 6 behaviours",
             loc="left", fontsize=8.5, pad=26)

fig.text(0.02, 0.01,
         "Hard = if-rules. Soft = TCN (GPU). Hard+Soft = max(hard acc, soft acc). "
         "Accuracy is class-imbalance inflated for mismatched behaviours; AUROC is the honest metric.",
         fontsize=6, color=C["grey"])
fig.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, "figE_六行为_硬准确率.png", outdir=os.path.join(HERE, "figs"))
print("saved figE_六行为_硬准确率.png (硬/软/硬加软 三栏)")
