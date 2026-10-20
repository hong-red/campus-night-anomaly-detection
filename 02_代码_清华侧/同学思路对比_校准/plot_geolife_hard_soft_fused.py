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
"""GeoLife（清华侧）六行为 × 三栏（硬/软/硬加软）准确率对比图。

硬     = if 规则的均衡准确率（真实准确率，避免运动域 0.82 的类别不平衡地板）
软     = 同学 GPU 的 TCN S_dl≥0.5 acc@0.5（tcn_gpu.json）
硬加软 = max(硬, 软)
风格与 sdd_hard_soft_fused_accuracy.png 一致（英文、蓝/橙/紫、柱顶数值、无红线、图例无后缀）。
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
hard = json.load(open(os.path.join(HERE, "hard_accuracy_6behaviors.json"), encoding="utf-8"))
soft = json.load(open(os.path.join(HERE, "tcn_gpu.json"), encoding="utf-8"))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering", "dwell": "Lingering",
         "wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag"}


def bal_acc(row):
    """均衡准确率 = (召回率 + 特异度)/2，消除类别不平衡地板。从 tp/fp/fn/neg_frac 反推 tn。"""
    tp, fp, fn, nf = row["tp"], row["fp"], row["fn"], row["neg_frac"]
    tn = (nf * (tp + fp + fn) - fp) / (1 - nf) if nf < 1 else float("inf")
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


hard_v = [bal_acc(hard[k]) for k in ORDER]
soft_v = [soft[k]["acc"] for k in ORDER]
fused_v = [max(h, s) for h, s in zip(hard_v, soft_v)]   # 硬加软 = max(硬, 软)

COLOR = {"hard": "#2C7BB6", "soft": "#EE7733", "fused": "#762A83"}

fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = np.arange(len(ORDER))
w = 0.24

ax.bar(x - w, hard_v, w, color=COLOR["hard"], label="Hard", zorder=3)
ax.bar(x, soft_v, w, color=COLOR["soft"], label="Soft", zorder=3)
ax.bar(x + w, fused_v, w, color=COLOR["fused"], label="Hard+Soft", zorder=3)

for xi, v in enumerate(hard_v):
    ax.text(xi - w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
for xi, v in enumerate(soft_v):
    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
for xi, v in enumerate(fused_v):
    ax.text(xi + w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=10)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)

ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10),
          ncol=3, fontsize=9.5, handlelength=1.4, columnspacing=1.3)

ax.set_title("GeoLife (Tsinghua) × Hard / Soft / Hard+Soft Accuracy", fontsize=13, pad=30)

fig.text(0.01, 0.012,
         "Hard = balanced accuracy (motion-domain rules are blind = 0.50, "
         "not the 0.82 class-imbalance floor); Soft = acc@0.5; Hard+Soft = max(hard, soft).",
         fontsize=8.5, color="#6E6E6E")

fig.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, "geolife_hard_soft_fused_accuracy.png", outdir=os.path.join(HERE, "figs"))
print("done")
