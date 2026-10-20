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
"""重画「六行为 × 硬准确率」图：图例移到图上缘之上，避免压到 1.00 标注/柱子。
数据直接读 hard_accuracy_6behaviors.json，不再重跑。
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
data = json.load(open(os.path.join(HERE, "hard_accuracy_6behaviors.json"), encoding="utf-8"))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering", "dwell": "Lingering",
         "wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag"}
DOM = {"geofence": "rule", "gather": "rule", "dwell": "rule",
       "wandering": "motion", "sprint": "motion", "zigzag": "motion"}

labels = [LABEL[k] for k in ORDER]
hard = [data[k]["acc"] for k in ORDER]
neg_frac = [data[k]["neg_frac"] for k in ORDER]
doms = [DOM[k] for k in ORDER]

fig, ax = plt.subplots(figsize=(DOUBLE_COL, 60 * MM))
x = np.arange(len(labels))
w = 0.26
soft = [0.0] * len(labels)
fused = [0.0] * len(labels)

ax.bar(x - w, hard, w, color=BRANCH_COLOR["rules"], label="Hard (if-rules)", zorder=3)
ax.bar(x, soft, w, color=BRANCH_COLOR["dl"], label="Soft (TCN) — pending", zorder=3)
ax.bar(x + w, fused, w, color=BRANCH_COLOR["fused"], label="Hard+Soft — pending", zorder=3)

for xi, (v, nf, dom) in enumerate(zip(hard, neg_frac, doms)):
    ax.text(xi - w, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)
    if dom == "motion":
        ax.plot([xi - w - 0.17, xi - w + 0.17], [nf, nf], color=C["alert"],
                lw=1.1, ls="--", zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Accuracy (acc@0.5)")
ax.set_ylim(0, 1.16)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.5)

# 图例移到图上缘之上，不压柱子
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.03),
          ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=1.2)

ax.set_title("Tsinghua (Geolife) — Hard branch accuracy across 6 behaviours",
             loc="left", fontsize=8.5, pad=26)

fig.text(0.02, 0.01,
         "Dashed red = 'always-normal' floor (class-imbalance; hard rule cannot see motion "
         "behaviours). Intrusion = 1.0 is exact by definition (rule == spatial test). "
         "Soft / Hard+Soft to be filled with GPU TCN results.",
         fontsize=6, color=C["grey"])
fig.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, "figE_六行为_硬准确率.png", outdir=os.path.join(HERE, "figs"))
print("saved figE with legend moved up")
