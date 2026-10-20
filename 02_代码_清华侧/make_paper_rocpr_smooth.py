#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成论文配图（清华侧 EVT/FluxEV 分支）：
  fig1_roc_pr.png          ROC(AUROC) + PR(AP)  —— eval 测试段口径
  fig4_smoothing_stages.png FluxEV 逐层平滑（真实清华轨迹 035，标 train|test）
  fig_rocpr_smooth.png      上面两张图的纵向拼接（对应论文里 fig_rocpr / fig_smooth）

与 F:\\仿真跑\\make_nature_figs.py 的 Fig1 / Fig4 一致，仅路径改为本仓库自包含 _paths.py。
风格：字体整体加粗、线条加粗；标题不使用破折号/等号分隔。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # 包根目录，_paths.py 所在

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import metrics

import _paths as _PKGP
from geolife_pipeline import (load_geofence_polygon, parse_plt,
                              longest_continuous_segment, resample, run_pipeline)

# 输出目录：与已有论文配图放在一起
OUT = os.path.join(_PKGP.ROOT, "10_论文配图", "清华侧")
os.makedirs(OUT, exist_ok=True)

# --------------------------- Nature 风格（加粗加厚版） ---------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "font.weight": "bold",              # 全局字体加粗
    "axes.labelsize": 7,
    "axes.labelweight": "bold",
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.9,              # 脊线加粗
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.4,             # 曲线加粗
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 350,
    "savefig.dpi": 350,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
MM = 1 / 25.4
C = {"alert": "#B2182B", "calm": "#4393C3", "accent": "#D6604D",
     "grey": "#6E6E6E", "dark": "#252525", "green": "#1B7837"}


def panel_label(ax, s, dx=-0.10, dy=1.08):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="right")


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)
    return p


# =====================================================================
# Fig 1  ROC / PR  (eval 模式, 基于 geolife_scores.csv)
# =====================================================================
csv_path = os.path.join(_PKGP.CODE_TSINGHUA, "results", "geolife_scores.csv")
ev = pd.read_csv(csv_path)

# 评估口径与管线一致：只用测试段(训练段之后)
n_ev = len(ev)
_test = np.arange(n_ev) >= n_ev // 2
g = ev["gt_label"].values[_test]
s = ev["final_score"].values[_test]
fpr, tpr, _ = metrics.roc_curve(g, s)
prec, rec, _ = metrics.precision_recall_curve(g, s)
auroc = metrics.roc_auc_score(g, s)
ap = metrics.average_precision_score(g, s)
print(f"[fig1] n_test={_test.sum()}  AUROC={auroc:.3f}  AP={ap:.3f}")

fig = plt.figure(figsize=(89 * MM, 54 * MM))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42, top=0.78)
ax = fig.add_subplot(gs[0, 0])
ax.plot(fpr, tpr, color=C["alert"], lw=2.0)
ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=C["grey"], lw=1.0)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"ROC (AUROC {auroc:.3f})", pad=3)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(rec, prec, color=C["calm"], lw=2.0)
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title(f"PR (AP {ap:.3f})", pad=3)
ax2.set_xlim(-0.01, 1.01); ax2.set_ylim(-0.01, 1.01)

fig.suptitle("EVT-branch ROC and PR on the Geolife test segment",
             fontsize=8, fontweight="bold", y=0.97)
fig1_path = save(fig, "fig1_roc_pr.png")

# =====================================================================
# Fig 4  FluxEV 逐层平滑 (真实清华轨迹)
# =====================================================================
poly = load_geofence_polygon(os.path.join(_PKGP.ROOT, "09_电子围栏", "geofence_tsinghua.json"))
real_p = os.path.join(_PKGP.GEOLIFE_DATA, "035", "Trajectory", "20090412205936.plt")
rlats, rlons, rt = parse_plt(real_p)
rlats, rlons, rt = longest_continuous_segment(rlats, rlons, rt)
dt = 5.0
rlat, rlon, rtg = resample(rlats, rlons, rt, dt)
res = run_pipeline(rlat, rlon, rtg, dt, poly=poly)
st = res["feat_stages"]["speed"]
tl = res["train_len"]

fig = plt.figure(figsize=(183 * MM, 96 * MM))
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)
rows = [("raw", "Speed (m s$^{-1}$)", C["dark"], "a"),
        ("pred_err", "EWMA pred. error", C["calm"], "b"),
        ("fs_err", "First smoothing", C["accent"], "c"),
        ("ss_err", "Second smoothing", C["alert"], "d")]
axes = []
for i, (key, lab, c, tag) in enumerate(rows):
    ax = fig.add_subplot(gs[i], sharex=axes[0] if axes else None)
    ax.plot(rtg, st[key], color=c, lw=1.3)
    ax.set_ylabel(lab, fontsize=6.0, fontweight="bold")
    ax.axvline(rtg[tl], ls=(0, (2, 2)), color=C["grey"], lw=1.0)
    panel_label(ax, tag)
    axes.append(ax)
axes[-1].set_xlabel("Time (s)")
axes[0].text(rtg[tl] * 1.01, axes[0].get_ylim()[1] * 0.92, "train | test",
             fontsize=6.5, color=C["grey"], fontweight="bold")
axes[0].set_title("FluxEV fluctuation extraction, layer by layer", pad=4)
fig4_path = save(fig, "fig4_smoothing_stages.png")

# =====================================================================
# 纵向拼接：对应论文里的 fig_rocpr / fig_smooth 两段式
# =====================================================================
from PIL import Image

GAP = 24
TARGET_W = 1800


def load_rgb(p):
    return Image.open(p).convert("RGB")


img1 = load_rgb(fig1_path)
img4 = load_rgb(fig4_path)

w1 = int(TARGET_W)
h1 = int(round(img1.height * TARGET_W / img1.width))
h4 = int(round(img4.height * TARGET_W / img4.width))
img1 = img1.resize((w1, h1), Image.LANCZOS)
img4 = img4.resize((w1, h4), Image.LANCZOS)

combined = Image.new("RGB", (w1, h1 + GAP + h4), (255, 255, 255))
combined.paste(img1, (0, 0))
combined.paste(img4, (0, h1 + GAP))
comb_path = os.path.join(OUT, "fig_rocpr_smooth.png")
combined.save(comb_path)
print("saved", comb_path, f"({combined.size[0]}x{combined.size[1]})")

print("\n完成：已重新生成 fig1_roc_pr.png / fig4_smoothing_stages.png / fig_rocpr_smooth.png")
