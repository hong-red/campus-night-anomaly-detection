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
"""同学思路 vs 当前项目 —— 两张对比图

图A  规则域 3 行为「仅 if 规则 vs + EVT 极值」双数据集对比（校准核心）
图B  六行为 × 4 检测器全景消融，清华 Geolife / 斯坦福 SDD 并排

数据来源（读 JSON，不硬编码）：
  清华 Geolife : F:\\仿真跑\\tcn_dl_branch\\results\\fusion_ablation.json
  斯坦福 SDD   : F:\\Stanford_Drone\\results\\sdd_fusion_ablation.json

输出：本目录 figs/
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKGP.CODE_TSINGHUA)
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib.pyplot as plt
from style import (apply_cn_style, C, BRANCH_COLOR, DOUBLE_COL, MM,
                   panel_label, save, chance_line)

apply_cn_style()

GEO = json.load(open(_PKGP.TCN_DIR + "/results/fusion_ablation.json", encoding="utf-8"))
SDD = json.load(open(_PKGP.CODE_STANFORD + "/results/sdd_fusion_ablation.json", encoding="utf-8"))
OUTDIR = os.path.join(_HERE, "figs")

# 同学的 6 行为（规则域 3 + 运动域 3）
BEHAVIORS = [
    ("geofence",  "禁区闯入"),
    ("gather",    "多人聚集"),
    ("dwell",     "原地逗留"),
    ("wandering", "徘徊"),
    ("sprint",    "疾跑"),
    ("zigzag",    "之字形"),
]
RULE_BEHAVIORS = BEHAVIORS[:3]

DET_ORDER = ["fluxev", "dl", "fused", "rules"]
DET_LABEL = {
    "fluxev": "FluxEV（规则+EVT 极值）",
    "dl":     "TCN（神经网络）",
    "fused":  "融合 max（规则+TCN）",
    "rules":  "仅 if 规则",
}


def _delta_text(delta):
    if delta > 0.003:
        return f"↑{delta:+.2f}"
    if delta < -0.003:
        return f"↓{delta:+.2f}"
    return "持平"


# --------------------------- 图A：规则域「仅 if 规则 vs + EVT 极值」 ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 50 * MM), sharey=True)
for ax, (key, zh) in zip(axes, RULE_BEHAVIORS):
    g_r, g_e = GEO[key]["rules"], GEO[key]["fluxev"]
    s_r, s_e = SDD[key]["rules"], SDD[key]["fluxev"]
    x = np.arange(2)               # 清华 / 斯坦福
    w = 0.34
    ax.bar(x - w / 2, [g_r, s_r], w, color=BRANCH_COLOR["rules"],
           label="仅 if 规则", zorder=3)
    ax.bar(x + w / 2, [g_e, s_e], w, color=BRANCH_COLOR["fluxev"],
           label="+ EVT 极值", zorder=3)

    for xi, (r, e) in zip(x, [(g_r, g_e), (s_r, s_e)]):
        ax.text(xi - w / 2, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + w / 2, e + 0.02, f"{e:.2f}", ha="center", va="bottom", fontsize=7.5)
        d = e - r
        col = C["alert"] if d < -0.003 else (C["green"] if d > 0.003 else C["grey"])
        ax.text(xi, max(r, e) + 0.10, _delta_text(d), ha="center", va="bottom",
                fontsize=7.5, color=col, fontweight="bold")

    chance_line(ax, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["清华 Geolife", "斯坦福 SDD"])
    ax.set_title(zh, pad=6)
    ax.set_ylim(0, 1.22)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)

axes[0].set_ylabel("AUROC")
axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
for ax, t in zip(axes, "abc"):
    panel_label(ax, t)

fig.suptitle("规则域 3 行为：加 EVT 极值是否有增益（同学判断的校准）",
             fontsize=13, y=1.00)
fig.tight_layout()
save(fig, "figA_规则域_加EVT极值对比.png", outdir=OUTDIR)


# --------------------------- 图B：六行为 × 4 检测器全景 ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 66 * MM), sharey=True)
for ax, (name, data) in zip(axes, [("清华 Geolife", GEO), ("斯坦福 SDD", SDD)]):
    x = np.arange(len(BEHAVIORS))
    w = 0.19
    for j, d in enumerate(DET_ORDER):
        vals = [data[k][d] for k, _ in BEHAVIORS]
        ax.bar(x - 1.5 * w + j * w, vals, w, color=BRANCH_COLOR[d],
               label=DET_LABEL[d], zorder=3)
        for xi, v in zip(x - 1.5 * w + j * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=5.4, rotation=90)

    chance_line(ax, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([zh for _, zh in BEHAVIORS])
    ax.set_title(name, pad=6)
    ax.set_ylim(0, 1.20)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)

axes[0].set_ylabel("AUROC")
axes[0].legend(frameon=False, loc="upper left", ncol=1, fontsize=8.5)
for ax, t in zip(axes, "ab"):
    panel_label(ax, t)

fig.suptitle("六行为全景消融：规则域靠 if 规则 / 运动域靠 TCN / 融合覆盖全部",
             fontsize=13, y=1.00)
fig.tight_layout()
save(fig, "figB_六行为_双数据集消融.png", outdir=OUTDIR)

print("done")
