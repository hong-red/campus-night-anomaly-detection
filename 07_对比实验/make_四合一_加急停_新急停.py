#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_四合一_evt_tcn.png 重制版：每个子图加小标题 + 加入第 7 类行为「急停(Sudden stop)」。

2×2 四宫格：
  左上  清华 Geolife  准确率（EVT=均衡准确率 / TCN=acc@0.5）
  右上  清华 Geolife  AUROC
  左下  斯坦福 SDD    准确率
  右下  斯坦福 SDD    AUROC

急停数据来自新重跑的 eval_sudden_stop_tsinghua.py / eval_sudden_stop_sdd.py
（逐帧、测试段>=n//2、多轨迹池化，与 6 行为同口径）。
图中「EVT」栏 = FluxEV 完整硬性分支 = max(EVT 运动极值, if 规则) = res["final"]。
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_cur = _HERE
for _i in range(6):
    if os.path.exists(os.path.join(_cur, "_paths.py")):
        sys.path.insert(0, _cur)
        break
    _cur = os.path.dirname(_cur)
import _paths as _PKGP

_ABL = os.path.join(_PKGP.ROOT, "06_消融实验", "结果_JSON")
_SDD = os.path.join(_PKGP.CODE_STANFORD, "results")
_NEW = os.path.join(_HERE, "结果_JSON")
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import apply_cn_style, save

apply_cn_style()
# 字体加粗加厚：微软雅黑 + 全量 bold（与 plot_final_3cols_*.py 同款）
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
for _k in ("font.size", "axes.labelsize", "axes.titlesize",
           "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
    plt.rcParams[_k] *= 2

# ---------------- 数据源 ----------------
hard = json.load(open(os.path.join(_ABL, "hard_accuracy_6behaviors.json"), encoding="utf-8"))
final3 = json.load(open(os.path.join(_ABL, "final_3cols_local.json"), encoding="utf-8"))
sdd_fw = json.load(open(os.path.join(_SDD, "sdd_fusion_framewise.json"), encoding="utf-8"))
ss_geo = json.load(open(os.path.join(_NEW, "geo_sudden_stop_50.json"), encoding="utf-8"))
ss_sdd = json.load(open(os.path.join(_NEW, "sdd_sudden_stop_3978.json"), encoding="utf-8"))

ORDER = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag", "sudden-stop"]
LABEL = {"geofence": "Intrusion", "gather": "Gathering", "dwell": "Lingering",
         "wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag",
         "sudden-stop": "Sudden stop"}
SUDDEN_NOTE = "Sudden stop: Geolife 50 traj / 17,573 frames; SDD 3,978 frames."


def bal_acc_negfrac(row):
    """均衡准确率，用 neg_frac 反推 tn（清华 hard_accuracy_6behaviors.json 口径）。"""
    tp, fp, fn, nf = row["tp"], row["fp"], row["fn"], row["neg_frac"]
    tn = (nf * (tp + fp + fn) - fp) / (1 - nf) if nf < 1 else float("inf")
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


def bal_acc_tn(row):
    """均衡准确率，直接用 tp/fp/fn/tn（SDD sdd_fusion_framewise.json 口径）。"""
    tp, fp, fn, tn = row["tp"], row["fp"], row["fn"], row["tn"]
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


# ---------------- 组装每个面板的 (EVT, TCN, EVT+TCN) ----------------
def panel_geo_acc():
    evt, tcn = [], []
    for k in ORDER:
        if k == "sudden-stop":
            e = ss_geo["fluxev"]["bal_acc"]
            t = ss_geo["dl"]["acc"]
        else:
            e = bal_acc_negfrac(hard[k])
            t = final3[k]["soft"]["acc"]
        evt.append(e)
        tcn.append(t)
    return evt, tcn


def panel_geo_auroc():
    evt, tcn = [], []
    for k in ORDER:
        if k == "sudden-stop":
            e = ss_geo["fluxev"]["auroc"]
            t = ss_geo["dl"]["auroc"]
        else:
            e = final3[k]["hard"]["auroc"]
            t = final3[k]["soft"]["auroc"]
        evt.append(e)
        tcn.append(t)
    return evt, tcn


def panel_sdd_acc():
    evt, tcn = [], []
    for k in ORDER:
        if k == "sudden-stop":
            e = ss_sdd["fluxev"]["bal_acc"]
            t = ss_sdd["dl"]["acc"]
        else:
            e = bal_acc_tn(sdd_fw[k]["rules"])
            t = sdd_fw[k]["dl"]["acc"]
        evt.append(e)
        tcn.append(t)
    return evt, tcn


def panel_sdd_auroc():
    evt, tcn = [], []
    for k in ORDER:
        if k == "sudden-stop":
            e = ss_sdd["fluxev"]["auroc"]
            t = ss_sdd["dl"]["auroc"]
        else:
            e = sdd_fw[k]["rules"]["auroc"]
            t = sdd_fw[k]["dl"]["auroc"]
        evt.append(e)
        tcn.append(t)
    return evt, tcn


COLOR = {"evt": "#2C7BB6", "tcn": "#EE7733", "fused": "#762A83"}

# ---------------- 绘图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(23.0, 12.5))
panels = [
    (axes[0, 0], "Tsinghua Geolife — Accuracy", "Accuracy", panel_geo_acc),
    (axes[0, 1], "Tsinghua Geolife — AUROC", "AUROC", panel_geo_auroc),
    (axes[1, 0], "Stanford Drone — Accuracy", "Accuracy", panel_sdd_acc),
    (axes[1, 1], "Stanford Drone — AUROC", "AUROC", panel_sdd_auroc),
]

x = np.arange(len(ORDER))
w = 0.30

for ax, title, ylabel, fn in panels:
    evt, tcn = fn()
    fused = [max(e, t) for e, t in zip(evt, tcn)]

    ax.bar(x - w, evt, w, color=COLOR["evt"], label="EVT", zorder=3)
    ax.bar(x, tcn, w, color=COLOR["tcn"], label="TCN", zorder=3)
    ax.bar(x + w, fused, w, color=COLOR["fused"], label="EVT+TCN", zorder=3)

    for xi, v in enumerate(evt):
        ax.text(xi - w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5)
    for xi, v in enumerate(tcn):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5)
    for xi, v in enumerate(fused):
        ax.text(xi + w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=14)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.9, 6.9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_title(title, fontsize=21, pad=10)

# 总标题（大标题）
fig.suptitle("EVT vs TCN vs EVT+TCN across 7 behaviours",
             fontsize=27, fontweight="bold", y=0.995)

# 共用图例（总标题下方、顶部居中）
handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR["evt"]),
    plt.Rectangle((0, 0), 1, 1, color=COLOR["tcn"]),
    plt.Rectangle((0, 0), 1, 1, color=COLOR["fused"]),
]
fig.legend(handles, ["EVT", "TCN", "EVT+TCN"], frameon=False,
           loc="upper center", ncol=3, fontsize=19, handlelength=1.4,
           columnspacing=1.3, bbox_to_anchor=(0.5, 0.965))

fig.text(0.5, 0.012, SUDDEN_NOTE, ha="center", va="bottom", fontsize=12)
fig.tight_layout(rect=[0, 0.035, 1, 0.92])
out = os.environ.get("FOURPANEL_OUT", r"F:\双分支实验_完整交付\07_对比实验\图")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_四合一_evt_tcn_新急停.png"), dpi=210, bbox_inches="tight")
print("saved fig_四合一_evt_tcn.png (加小标题 + 急停)")
