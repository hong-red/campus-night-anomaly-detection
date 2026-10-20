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
"""FluxEV 算法示意图（重做）：体现真实实现，并突出「急停/降速」这一亮点。

改动两处（对应反馈）：
  1. 异常场景从「疾跑爆发(加速)」改成「急停/降速」——这是 FluxEV EVT 运动分支的
     独有亮点（dwell 规则要停满 5 分钟才报，急停只能靠 FluxEV 抓）。
  2. 五步流水线不再用「滑动均值/窗口max/99分位」的示意近似，而是**直接调用
     geolife_pipeline.fluxev_score()**，画出它的真实中间阶段：
        raw → EWMA 预测误差 → 一阶平滑(std 增量=flux) → 二阶平滑(峰保持) → GPD 分数
     保证图与实际代码逐阶段一致。
"""

import os
import sys
sys.path.insert(0, _PKGP.CODE_TSINGHUA)
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import apply_cn_style, DOUBLE_COL, MM, save
import geolife_pipeline as gp

apply_cn_style()
# 字体加粗加大：改用微软雅黑(真粗体)并整体放大 1.5x
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
for _k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
    plt.rcParams[_k] *= 1.5
HERE = os.path.dirname(os.path.abspath(__file__))

RED, INK, GREY, BLUE = "#E8272C", "#252525", "#9a9a9a", "#0B6BFF"

# ---- 合成信号：正常走 + 一段急停/降速（速度骤降到 0.1） ----
n = 560
train_len = n // 2                 # 前一半给 SPOT 做 GPD 标定
t = np.arange(n)
rng = np.random.default_rng(1)
speed = np.full(n, 1.4) + rng.normal(0, 0.06, n)          # 正常行走 ~1.4 m/s
i0, i1 = 340, 400                                          # 测试段内急停
speed[i0:i1] = 0.1 + rng.normal(0, 0.05, i1 - i0)          # 骤降到 ~0.1（急停）
anom = (t >= i0) & (t < i1)

# ---- 直接跑项目里的真实 FluxEV，取它的中间阶段 ----
scores, feat, th, stages = gp.fluxev_score(speed, train_len, period=30,
                                           s_w=10, p_w=5, half_d_w=2, q=0.005)
pred_err = stages["pred_err"]      # EWMA 预测误差
fs_err = stages["fs_err"]          # 一阶平滑：std 增量（flux 波动）
ss_err = stages["ss_err"]          # 二阶平滑：峰保持
t_test = t[train_len:]             # 分数只覆盖测试段

panels = [
    ("1. raw speed", "m/s", t, speed, BLUE),
    ("2. EWMA error", "|err|", t, np.abs(pred_err), INK),
    ("3. flux (std↑)", "std increase", t, fs_err, INK),
    ("4. peak hold", "peak", t, ss_err, INK),
    ("5. GPD score", "0–1", t_test, scores, RED),
]

fig, axes = plt.subplots(1, 5, figsize=(DOUBLE_COL, 44 * MM))
for ax, (title, ylab, xx, s, color) in zip(axes, panels):
    ax.plot(xx, s, lw=1.5, color=color)
    ax.axvspan(i0, i1, color=RED, alpha=0.12, lw=0)
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.set_xlabel(ylab, fontsize=11)

# 分数面板加报警阈值线
axes[-1].axhline(0.5, color=RED, ls="--", lw=1.0)
axes[-1].text(1, 0.53, "alarm 0.5", transform=axes[-1].transAxes,
              fontsize=10, color=RED, ha="right", va="bottom")

# 面板间箭头
for i in range(4):
    fig.text((i + 1.08) / 5 + 0.005, 0.50, "→", fontsize=22, color=GREY, ha="center", va="center")

fig.suptitle("FluxEV — detecting sudden deceleration", fontsize=17, y=0.98, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig_fluxev_algorithm.png", outdir=os.path.join(HERE, "figs"))
print("done")
