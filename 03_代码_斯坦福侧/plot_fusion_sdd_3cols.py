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
"""SDD 六行为 × 三栏（硬 / 软 / 硬加软）准确率对比图。

硬     = max(dwell, gather, geofence) if 规则 的均衡准确率（真实准确率，见 bal_acc）
软     = 同学 GPU 跑的 S_dl≥0.5 acc@0.5（用户提供的数值，非本地 checkpoint）
硬加软 = max(硬, 软)  —— 软用同学 GPU 的数值

纵轴：准确率。硬用「均衡准确率」避免运动域普通准确率 0.78 的类别不平衡地板假象（硬真盲=0.50）；
软用同学提供的 acc@0.5。虚线=chance 0.5。
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
RES = json.load(open(os.path.join(HERE, "results", "sdd_fusion_framewise.json"), encoding="utf-8"))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering", "dwell": "Lingering",
         "wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag"}
DOM = {"geofence": "rule", "gather": "rule", "dwell": "rule",
       "wandering": "motion", "sprint": "motion", "zigzag": "motion"}

# 软 = 同学 GPU 跑的 S_dl（用户提供；同学改了 TCN 参数 + GPU，比本地好）
SOFT_USER = {"geofence": 0.306, "gather": 0.639, "dwell": 0.453,
             "wandering": 0.630, "sprint": 0.577, "zigzag": 0.786}

def bal_acc(row):
    """均衡准确率 = (召回率 + 特异度)/2，消除类别不平衡地板（真实准确率）。"""
    tp, fp, fn, tn = row["tp"], row["fp"], row["fn"], row["tn"]
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2

hard = [bal_acc(RES[k]["rules"]) for k in ORDER]
soft = [SOFT_USER[k] for k in ORDER]
fused = [max(h, s) for h, s in zip(hard, soft)]   # 硬加软 = max(硬, 同学软)

COLOR = {"hard": "#2C7BB6", "soft": "#EE7733", "fused": "#762A83"}

fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = np.arange(len(ORDER))
w = 0.24

ax.bar(x - w, hard, w, color=COLOR["hard"], label="Hard", zorder=3)
ax.bar(x, soft, w, color=COLOR["soft"], label="Soft", zorder=3)
ax.bar(x + w, fused, w, color=COLOR["fused"], label="Hard+Soft", zorder=3)

for xi, v in enumerate(hard):
    ax.text(xi - w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
for xi, v in enumerate(soft):
    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
for xi, v in enumerate(fused):
    ax.text(xi + w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=10)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)

ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10),
          ncol=4, fontsize=9.5, handlelength=1.4, columnspacing=1.3)

ax.set_title("Stanford Drone (SDD) × Hard / Soft / Hard+Soft Accuracy", fontsize=13, pad=30)

fig.text(0.01, 0.012,
         "Hard = balanced accuracy (motion-domain rules are blind = 0.50, "
         "not the 0.78 class-imbalance floor); Soft = acc@0.5; Hard+Soft = max(hard, soft).",
         fontsize=8.5, color="#6E6E6E")

fig.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, "sdd_hard_soft_fused_accuracy.png", outdir=os.path.join(HERE, "results"))
print("done")
