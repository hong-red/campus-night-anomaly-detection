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
"""六行为 × 三栏(硬/软/硬加软) 区分度(AUROC) —— 比照准确度图版式。

硬     = if 规则 max(dwell[lo=15s], gather, geofence) 的 AUROC
软     = 本地 TCN z-clip 的 AUROC
硬加软 = 逐帧 max(硬,软) 的 AUROC
数据源：final_3cols_local.json（同一口径，第二半段，50 条行走轨迹）。
AUROC = 0.5 即无区分度(随机)。
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
# 字体加粗加厚：改用微软雅黑(有真粗体)并全量 bold
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
for _k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
    plt.rcParams[_k] *= 2

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "final_3cols_local.json"), encoding="utf-8"))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering", "dwell": "Lingering",
         "wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag"}
GROUP = {"geofence": "if-else", "gather": "if-else", "dwell": "if-else+FluxEV",
         "wandering": "TCN", "sprint": "TCN", "zigzag": "TCN"}
GROUP_COLOR = {"if-else": "#1B7837", "if-else+FluxEV": "#4393C3", "TCN": "#eb6834"}

hard_v = [RES[k]["hard"]["auroc"] for k in ORDER]
soft_v = [RES[k]["soft"]["auroc"] for k in ORDER]
fused_v = [max(h, s) for h, s in zip(hard_v, soft_v)]   # 取 max（与准确度图同口径）

for k, h, s, f in zip(ORDER, hard_v, soft_v, fused_v):
    print(f"{k:<10} hard={h:.3f}  soft={s:.3f}  fused={f:.3f}")

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
ax.set_ylabel("AUROC", fontsize=22)
ax.set_ylim(0, 1.28)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(-0.9, 5.9)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)

ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04),
          ncol=3, fontsize=19, handlelength=1.4, columnspacing=1.3)

ax.set_title("GeoLife EVT vs TCN vs EVT+TCN (AUROC)",
             fontsize=26, pad=8)



fig.tight_layout(rect=[0, 0.045, 1, 1])
save(fig, "fig_六行为_硬软硬加软_区分度.png", outdir=os.path.join(HERE, "figs"))
print("saved figs/fig_六行为_硬软硬加软_区分度.png")
