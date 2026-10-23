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
"""四合一汇总图：GeoLife / Stanford Drone × Accuracy / AUROC（2×2）。

把四张单图拼成一张：
  (0,0) GeoLife 准确率     (0,1) GeoLife 区分度 AUROC
  (1,0) SDD 准确率         (1,1) SDD 区分度 AUROC

子图去掉各自图例，图下方放一个共用图例（EVT / TCN / EVT+TCN），
图上方一个总标题。
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import apply_cn_style, save

apply_cn_style()
# 字体加粗加厚：改用微软雅黑(有真粗体)并全量 bold
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

TS = _PKGP.CODE_TSINGHUA
SD = _PKGP.CODE_STANFORD

hard = json.load(open(os.path.join(TS, "同学思路对比_校准", "hard_accuracy_6behaviors.json"), encoding="utf-8"))
glocal = json.load(open(os.path.join(TS, "同学思路对比_校准", "final_3cols_local.json"), encoding="utf-8"))
sdd = json.load(open(os.path.join(SD, "results", "sdd_fusion_framewise.json"), encoding="utf-8"))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering", "dwell": "Lingering",
         "wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag"}


def bal_acc_geolife(row):
    """均衡准确率 = (召回率 + 特异度)/2（GeoLife 用 neg_frac 推 tn）。"""
    tp, fp, fn, nf = row["tp"], row["fp"], row["fn"], row["neg_frac"]
    tn = (nf * (tp + fp + fn) - fp) / (1 - nf) if nf < 1 else float("inf")
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


def bal_acc_sdd(row):
    """均衡准确率 = (召回率 + 特异度)/2（SDD 直接给 tn）。"""
    tp, fp, fn, tn = row["tp"], row["fp"], row["fn"], row["tn"]
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


# ---- 四个面板：(子图标题, 纵轴标签, EVT 值, TCN 值) ----
panels = {
    (0, 0): ("Accuracy",
             [bal_acc_geolife(hard[k]) for k in ORDER],
             [glocal[k]["soft"]["acc"] for k in ORDER]),
    (0, 1): ("AUROC",
             [glocal[k]["hard"]["auroc"] for k in ORDER],
             [glocal[k]["soft"]["auroc"] for k in ORDER]),
    (1, 0): ("Accuracy",
             [bal_acc_sdd(sdd[k]["rules"]) for k in ORDER],
             [sdd[k]["dl"]["acc"] for k in ORDER]),
    (1, 1): ("AUROC",
             [sdd[k]["rules"]["auroc"] for k in ORDER],
             [sdd[k]["dl"]["auroc"] for k in ORDER]),
}

COLOR = {"evt": "#2C7BB6", "tcn": "#EE7733", "fused": "#762A83"}

fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
x = np.arange(len(ORDER))
w = 0.33

for (r, c), (ylabel, evt, tcn) in panels.items():
    ax = axes[r, c]
    fused = [max(h, s) for h, s in zip(evt, tcn)]

    ax.bar(x - w, evt, w, color=COLOR["evt"], zorder=3)
    ax.bar(x, tcn, w, color=COLOR["tcn"], zorder=3)
    ax.bar(x + w, fused, w, color=COLOR["fused"], zorder=3)

    for xi, v in enumerate(evt):
        ax.text(xi - w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    for xi, v in enumerate(tcn):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    for xi, v in enumerate(fused):
        ax.text(xi + w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.9, 5.9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)

# 共用图例（图下方，一行三个）
handles = [Patch(facecolor=COLOR["evt"]), Patch(facecolor=COLOR["tcn"]),
           Patch(facecolor=COLOR["fused"])]
fig.legend(handles, ["EVT", "TCN", "EVT+TCN"], loc="lower center",
           bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=15,
           handlelength=1.4, columnspacing=1.5)

# 总标题
fig.suptitle("EVT vs TCN vs EVT+TCN on GeoLife and Stanford Drone",
             fontsize=20, fontweight="bold", y=0.94)

fig.tight_layout(rect=[0, 0.07, 1, 0.92])
save(fig, "fig_四合一_evt_tcn.png",
     outdir=os.path.join(_PKGP.ROOT, "08_实验结论", "最终图"))
print("saved 08_实验结论/最终图/fig_四合一_evt_tcn.png")
