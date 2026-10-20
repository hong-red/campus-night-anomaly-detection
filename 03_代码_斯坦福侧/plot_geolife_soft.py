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
"""GeoLife（清华侧）软分支 TCN —— 六行为 × 五指标 对比图。

数据来源：同学 GPU 跑的 S_dl≥0.5（帧级，GeoLife，用户提供）。
风格：与 sdd_hard_soft_fused_accuracy.png 一致（英文、柱顶数值、无红线、图例无后缀）。
"""

import os
import sys

sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import apply_cn_style, save

apply_cn_style()

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = ["Intrusion", "Gathering", "Lingering", "Wandering", "Sprint", "Zigzag"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC"]

# 同学 GeoLife 软分支（帧级，阈值 S_dl≥0.5）
DATA = {
    "Intrusion": [0.862, 0.030, 0.023, 0.026, 0.463],
    "Gathering": [0.812, 0.231, 0.092, 0.132, 0.610],
    "Lingering": [0.766, 0.078, 0.033, 0.046, 0.147],
    "Wandering": [0.769, 0.233, 0.106, 0.145, 0.567],
    "Sprint":    [0.836, 0.570, 0.470, 0.515, 0.770],
    "Zigzag":    [0.787, 0.280, 0.093, 0.140, 0.648],
}

COLOR = ["#2C7BB6", "#EE7733", "#762A83", "#1B7837", "#9AA0A6"]

fig, ax = plt.subplots(figsize=(11.0, 5.2))
x = np.arange(len(ORDER))
w = 0.16

for i, m in enumerate(METRICS):
    vals = [DATA[k][i] for k in ORDER]
    off = (i - (len(METRICS) - 1) / 2) * w
    ax.bar(x + off, vals, w, color=COLOR[i], label=m, zorder=3)
    for xi, v in enumerate(vals):
        ax.text(xi + off, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(ORDER, fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)

ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10),
          ncol=5, fontsize=9.5, handlelength=1.2, columnspacing=1.0)

ax.set_title("GeoLife (Tsinghua) × Soft branch (S_dl≥0.5) — 6 behaviors",
             fontsize=13, pad=30)

fig.tight_layout(rect=[0, 0.03, 1, 1])
save(fig, "geolife_soft_metrics.png", outdir=os.path.join(HERE, "results"))
print("done")
