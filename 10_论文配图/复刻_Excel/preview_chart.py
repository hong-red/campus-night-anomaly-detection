# -*- coding: utf-8 -*-
"""按 xlsx 原生图表的同一几何/配色渲染一张预览 PNG，便于视觉核对。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
data = json.load(open(os.path.join(ROOT, "06_消融实验", "结果_JSON",
                                   "cross_validate_schools_3beh.json"), encoding="utf-8"))

BEHAVIORS = [("wandering", "Wandering"), ("sprint", "Sprint"), ("zigzag", "Zigzag")]
COMBOS = [("geo_geo", "Geo->Geo (in)"), ("geo_sdd", "Geo->SDD (trans)"),
          ("sdd_sdd", "SDD->SDD (in)"), ("sdd_geo", "SDD->Geo (trans)")]
COLOR = {"geo_geo": "#9EC5F4", "geo_sdd": "#2A78D6",
         "sdd_sdd": "#F6C9A8", "sdd_geo": "#EB6834"}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]

fig, ax = plt.subplots(figsize=(10.5, 6.0))
y = np.arange(len(BEHAVIORS)); h = 0.19
for j, (ck, cname) in enumerate(COMBOS):
    off = 1.5 * h - j * h
    vals = [data[k][ck] for k, _ in BEHAVIORS]
    ax.barh(y + off, vals, h, color=COLOR[ck], zorder=3, label=cname)
    for yi, v in zip(y + off, vals):
        ax.text(v + 0.012, yi, f"{v:.2f}", ha="left", va="center", fontsize=11)

ax.set_yticks(y)
ax.set_yticklabels([en for _, en in BEHAVIORS], fontsize=15)
ax.set_xlabel("AUROC", fontsize=16)
ax.set_xlim(0, 1.12)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_axisbelow(True)
ax.grid(axis="x", color="#e6e6e6", lw=0.6)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=10)
ax.set_title("Frame-level cross-dataset generalization (AUROC)", fontsize=14, pad=30)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(os.path.join(HERE, "_preview_excel_chart.png"), dpi=130)
print("saved preview")
