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
"""六行为 × 三栏(硬/EVT/软) 准确率对比图 —— 硬、EVT 用真实数据，软留待 GPU TCN。

数据来源：
  硬  : hard_accuracy_6behaviors.json
  EVT : evt_motion_3behaviors.json (后3) + evt_motion_rule3.json (前3)
  软  : 待 GPU TCN，暂 0
纵轴：准确率 acc@0.5（用户要求）。运动类的硬/EVT 准确率≈类别不平衡地板，脚注标注。
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import apply_nature_style, C, DOMAIN_COLOR, DOUBLE_COL, MM, save
apply_nature_style()

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering", "dwell": "Lingering",
         "wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag"}
DOM = {"geofence": "rule", "gather": "rule", "dwell": "rule",
       "wandering": "motion", "sprint": "motion", "zigzag": "motion"}

hard = json.load(open(os.path.join(HERE, "hard_accuracy_6behaviors.json"), encoding="utf-8"))
evt3 = json.load(open(os.path.join(HERE, "evt_motion_3behaviors.json"), encoding="utf-8"))
evt_rule = json.load(open(os.path.join(HERE, "evt_motion_rule3.json"), encoding="utf-8"))
evt = {**evt_rule, **evt3}

labels = [LABEL[k] for k in ORDER]
hard_v = [hard[k]["acc"] for k in ORDER]
evt_v = [evt[k]["acc"] for k in ORDER]
soft_v = [0.0] * len(ORDER)
doms = [DOM[k] for k in ORDER]
neg_frac = [hard[k]["neg_frac"] for k in ORDER]

fig, ax = plt.subplots(figsize=(DOUBLE_COL, 62 * MM))
x = np.arange(len(labels))
w = 0.24

ax.bar(x - w, hard_v, w, color=DOMAIN_COLOR["rule"], label="Hard (if-rules)", zorder=3)
ax.bar(x, evt_v, w, color=DOMAIN_COLOR["evt"], label="EVT (motion extreme)", zorder=3)
ax.bar(x + w, soft_v, w, color=DOMAIN_COLOR["tcn"], label="Soft (TCN) — pending", zorder=3)

for xi, (v, dom) in enumerate(zip(hard_v, doms)):
    ax.text(xi - w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0)
for xi, v in enumerate(evt_v):
    ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0)

# 运动类的类别不平衡地板(全判正常=负例占比)标注
for xi, (dom, nf) in enumerate(zip(doms, neg_frac)):
    if dom == "motion":
        ax.plot([xi - 0.42, xi + 0.18], [nf, nf], color=C["alert"], lw=1.0, ls="--", zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Accuracy (acc@0.5)")
ax.set_ylim(0, 1.16)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.5)

ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.03),
          ncol=3, fontsize=6.3, handlelength=1.1, columnspacing=1.1)

ax.set_title("Tsinghua (Geolife) — accuracy across 6 behaviours (3 detectors)",
             loc="left", fontsize=8.5, pad=26)

fig.text(0.02, 0.01,
         "Dashed red = 'always-normal' floor: for motion behaviours the Hard/EVT accuracy is "
         "class-imbalance, NOT real detection (their AUROC is ~0.5). Soft (TCN) pending GPU.",
         fontsize=6, color=C["grey"])
fig.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, "figE_六行为_三栏准确率.png", outdir=os.path.join(HERE, "figs"))
print("saved figE_六行为_三栏准确率.png")
