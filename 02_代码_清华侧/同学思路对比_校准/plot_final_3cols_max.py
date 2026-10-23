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
"""六行为 × 三栏(硬 / 软 / 硬加软) 准确率 —— 本地软 + 取 max 版。

与 SDD 图 / plot_geolife_hard_soft_fused.py 同口径：
  硬     = if 规则 max(dwell[lo=15s], gather, geofence) 的「均衡准确率」bal_acc
          （(召回率+特异度)/2，消除运动域 0.82 的类别不平衡地板，运动域真盲=0.50）
  软     = 本地 TCN z-clip S_dl≥0.5 的 acc@0.5（final_3cols_local.json）
  硬加软 = max(硬准确率, 软准确率)   ← 用户要求「取 max」

分组（最终架构）：2 if-else / 1 if-else+FluxEV / 3 TCN。
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = _PKGP.CODE_TSINGHUA
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import apply_cn_style, save

apply_cn_style()
# 字体加粗加厚：改用微软雅黑(有真粗体)并全量 bold
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
for _k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
    plt.rcParams[_k] *= 2

HERE = os.path.dirname(os.path.abspath(__file__))
hard = json.load(open(os.path.join(HERE, "hard_accuracy_6behaviors.json"), encoding="utf-8"))
soft = json.load(open(os.path.join(HERE, "final_3cols_local.json"), encoding="utf-8"))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering",
         "dwell": "Lingering", "wandering": "Wandering",
         "sprint": "Sprint", "zigzag": "Zigzag"}
GROUP = {"geofence": "if-else", "gather": "if-else", "dwell": "if-else+FluxEV",
         "wandering": "TCN", "sprint": "TCN", "zigzag": "TCN"}
GROUP_CN = {"if-else": "if-else", "if-else+FluxEV": "if-else + FluxEV",
            "TCN": "TCN"}
GROUP_COLOR = {"if-else": "#1B7837", "if-else+FluxEV": "#4393C3", "TCN": "#eb6834"}


def bal_acc(row):
    """均衡准确率 = (召回率 + 特异度)/2，消除类别不平衡地板。"""
    tp, fp, fn, nf = row["tp"], row["fp"], row["fn"], row["neg_frac"]
    tn = (nf * (tp + fp + fn) - fp) / (1 - nf) if nf < 1 else float("inf")
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


hard_v = [bal_acc(hard[k]) for k in ORDER]
soft_v = [soft[k]["soft"]["acc"] for k in ORDER]
fused_v = [max(h, s) for h, s in zip(hard_v, soft_v)]   # 取 max

for k, h, s, f in zip(ORDER, hard_v, soft_v, fused_v):
    print(f"{k:<10} 硬(均衡)={h:.3f}  软={s:.3f}  硬加软(max)={f:.3f}")

COLOR = {"hard": "#2C7BB6", "soft": "#EE7733", "fused": "#762A83"}

fig, ax = plt.subplots(figsize=(11.5, 5.8))
x = np.arange(len(ORDER))
w = 0.32

ax.bar(x - w, hard_v, w, color=COLOR["hard"], label="EVT", zorder=3)
ax.bar(x, soft_v, w, color=COLOR["soft"], label="TCN", zorder=3)
ax.bar(x + w, fused_v, w, color=COLOR["fused"], label="EVT+TCN", zorder=3)

for xi, v in enumerate(hard_v):
    ax.text(xi - w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=15)
for xi, v in enumerate(soft_v):
    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=15)
for xi, v in enumerate(fused_v):
    ax.text(xi + w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=15)

ax.set_xticks(x)
ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=18)
ax.set_ylabel("Accuracy", fontsize=22)
ax.set_ylim(0, 1.28)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(-0.9, 5.9)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)

ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04),
          ncol=3, fontsize=19, handlelength=1.4, columnspacing=1.3)

ax.set_title("GeoLife EVT vs TCN vs EVT+TCN (accuracy)",
             fontsize=26, pad=8)



fig.tight_layout(rect=[0, 0.045, 1, 1])
save(fig, "fig_六行为_硬软硬加软_本地软.png", outdir=os.path.join(HERE, "figs"))
print("saved figs/fig_六行为_硬软硬加软_本地软.png")
