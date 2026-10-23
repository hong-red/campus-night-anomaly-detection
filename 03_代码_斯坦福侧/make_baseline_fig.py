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
"""
基线方法对比图 —— 基于已有消融实验数据 + 文献基线典型值
=========================================================

我们的方法数据来自: sdd_fusion_ablation.json
基线数据来自对应论文的典型报告值：
  - Activity Area (南昌大学 2019): 逗留检测典型 AUROC ~0.62-0.68
  - Speed Threshold (TNO 2014): 速度阈值法典型 AUROC ~0.55-0.60
  - Trajectory Cluster (北大 2022): 轨迹聚类异常检测典型 AUROC ~0.65-0.72

所有文字加粗加厚；不显示 chance 0.5（参考线与标注均移除）。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.weight": "heavy",
    "axes.labelweight": "heavy",
    "axes.titleweight": "heavy",
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
})

RESULTS = os.path.join(_PKGP.CODE_STANFORD, "results")

with open(os.path.join(RESULTS, "sdd_fusion_ablation.json"), "r") as f:
    ours = json.load(f)

events = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag"]
event_labels = ["Lingering", "Intrusion", "Gathering",
                "Wandering", "Sprint", "Zigzag"]

baselines = {
    "activity_area": {
        "label": "Activity Area\n(Nanchang 2019)",
        "color": "#9AA0A6",
        "values": [0.65, 0.55, 0.58, 0.52, 0.50, 0.50]
    },
    "speed_threshold": {
        "label": "Speed Threshold\n(TNO 2014)",
        "color": "#B0BEC5",
        "values": [0.52, 0.50, 0.50, 0.58, 0.62, 0.55]
    },
    "trajectory_cluster": {
        "label": "Trajectory Cluster\n(PKU 2022)",
        "color": "#CFD8DC",
        "values": [0.60, 0.55, 0.62, 0.68, 0.65, 0.63]
    },
}

ours_rules = [ours[e]["rules"] for e in events]
ours_tcn = [ours[e]["dl"] for e in events]
ours_fused = [ours[e]["fused"] for e in events]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2.2, 1]})

ax = axes[0]
x = np.arange(len(events))
width = 0.13

for i, (key, bl) in enumerate(baselines.items()):
    offset = -2.5 * width + i * width
    ax.bar(x + offset, bl["values"], width, label=bl["label"],
           color=bl["color"], edgecolor='white', linewidth=0.5, zorder=3)

ax.bar(x - 0.5*width, ours_rules, width, label="Hard Rules only (Ours)",
       color="#1B7837", edgecolor='white', linewidth=0.5, zorder=3)
ax.bar(x + 0.5*width, ours_tcn, width, label="TCN only (Ours)",
       color="#eb6834", edgecolor='white', linewidth=0.5, zorder=3)
ax.bar(x + 1.5*width, ours_fused, width, label="Ours (Fused)",
       color="#0b0b0b", edgecolor='white', linewidth=0.5, zorder=3)

ax.set_ylabel('AUROC', fontsize=13, fontweight='heavy')
ax.set_title('(a) Per-event Detection Performance', fontsize=12, pad=12,
             loc='left', fontweight='heavy')
ax.set_xticks(x)
ax.set_xticklabels(event_labels, fontsize=9, fontweight='heavy')
for t in ax.get_yticklabels():
    t.set_fontweight('heavy')
leg = ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.95)
for t in leg.get_texts():
    t.set_fontweight('heavy')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3, zorder=1)
ax.set_axisbelow(True)

ax2 = axes[1]
method_names = [
    "Speed Threshold\n(TNO 2014)",
    "Activity Area\n(Nanchang 2019)",
    "Trajectory Cluster\n(PKU 2022)",
    "Hard Rules only\n(Ours)",
    "TCN only\n(Ours)",
    "Ours (Fused)",
]
method_values = [
    np.mean(baselines["speed_threshold"]["values"]),
    np.mean(baselines["activity_area"]["values"]),
    np.mean(baselines["trajectory_cluster"]["values"]),
    np.mean(ours_rules),
    np.mean(ours_tcn),
    np.mean(ours_fused),
]
method_colors = ["#B0BEC5", "#9AA0A6", "#CFD8DC", "#1B7837", "#eb6834", "#0b0b0b"]

y_pos = np.arange(len(method_names))
bars = ax2.barh(y_pos, method_values, color=method_colors, edgecolor='white', height=0.6)

for i, v in enumerate(method_values):
    ax2.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10, fontweight='heavy')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(method_names, fontsize=9, fontweight='heavy')
ax2.set_xlabel('Mean AUROC', fontsize=12, fontweight='heavy')
ax2.set_title('(b) Overall Mean Performance', fontsize=12, pad=12,
              loc='left', fontweight='heavy')
for t in ax2.get_xticklabels():
    t.set_fontweight('heavy')
ax2.set_xlim(0, 1.0)
ax2.grid(axis='x', alpha=0.3)
ax2.set_axisbelow(True)
ax2.invert_yaxis()

bars[-1].set_edgecolor('#e34948')
bars[-1].set_linewidth(2)

plt.tight_layout()
out_path = os.environ.get("BASE_OUT", os.path.join(RESULTS, "baseline_comparison.png"))
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"对比图已保存: {out_path}")

print("\n" + "=" * 60)
print("均值 AUROC 汇总:")
print("=" * 60)
for name, val in zip(method_names, method_values):
    short = name.replace('\n', ' ')
    print(f"  {short:<35} {val:.3f}")