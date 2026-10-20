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
TCN 轨迹预测分支 —— 真实神经网络构成图（严格对应 tcn_branch.py 的 TCNPredictor）

结构事实(代码):
  input  (B, 16, 2) 相对坐标 -> permute -> (B, 2, 16)
  4 个残差块, dilation = 1,2,4,8, kernel=3, hidden=32
    每块: CausalConv1d -> ReLU -> Dropout -> CausalConv1d -> (+residual) -> ReLU
    残差: block1 用 1x1 Conv(2->32) 对齐通道; block2~4 恒等(identity)
  Flatten (32*16=512) -> Linear(512->64) -> ReLU -> Linear(64->16)
  view -> (B, 8, 2) 预测未来 8 帧
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import sdd_common as S

RESULTS = os.path.join(S.RESULTS_DIR, "paper_figs")
os.makedirs(RESULTS, exist_ok=True)

C_INK = S.C_INK
C_MUTED = S.C_MUTED
C_GRID = S.C_GRID
C_TCN = "#2a78d6"       # 卷积蓝
C_ACT = "#eb6834"       # 激活橙
C_HEAD = "#1baf7a"      # 全连接绿
C_RES = "#898781"       # 残差灰


def fig_tcn_architecture():
    fig, ax = plt.subplots(figsize=(10.0, 12.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 134); ax.axis("off")

    def bbox(x, y, w, h, title, sub, fc, ec, fs_t=8.2, fs_s=7.0, tc="white", tc_s="white"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                           fc=fc, ec=ec, lw=1.0)
        ax.add_patch(p)
        if sub:
            ax.text(x + w / 2, y + h / 2 + 0.9, title, ha="center", va="center",
                    fontsize=fs_t, color=tc, weight="bold")
            ax.text(x + w / 2, y + h / 2 - 1.2, sub, ha="center", va="center",
                    fontsize=fs_s, color=tc_s)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=fs_t, color=tc, weight="bold")

    def arr(x1, y1, x2, y2, color="0.4", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.2, color=color, linestyle=ls))

    # ---------- 输入 ----------
    bbox(30, 126, 40, 5.6, "Input: past trajectory",
         "(B, 16, 2) rel. coords  $\\to$  permute  $\\to$  (B, 2, 16)", "#f2f2ef", "0.55",
         tc=C_INK, tc_s=C_INK)
    arr(50, 125.6, 50, 121.5)

    # ---------- 4 个残差块 ----------
    dilations = [1, 2, 4, 8]
    block_top = 118.0
    block_h = 23.5
    for i, d in enumerate(dilations):
        y = block_top - i * block_h
        in_c = 2 if i == 0 else 32
        res_label = "1×1 Conv\n(2→32)" if i == 0 else "identity"
        # 块容器
        box = FancyBboxPatch((7, y - block_h + 3.5), 86, block_h - 4.5,
                             boxstyle="round,pad=0.4,rounding_size=1.2",
                             fc="#eef4fb", ec="#9ec5f4", lw=1.0)
        ax.add_patch(box)
        ax.text(9.5, y - 1.2, f"Residual block {i+1}   (dilation d={d}, k=3)",
                fontsize=8.6, color=C_TCN, weight="bold", va="center")

        # 主路径
        bx = 12
        bbox(bx, y - 6.5, 17, 6.5, f"CausalConv1d", f"in={in_c}→32", C_TCN, C_TCN)
        arr(bx + 17, y - 3.25, bx + 21, y - 3.25)
        bbox(bx + 21, y - 6.0, 12, 5.5, "ReLU\nDropout", "", C_ACT, C_ACT, fs_t=6.6)
        arr(bx + 33, y - 3.25, bx + 37, y - 3.25)
        bbox(bx + 37, y - 6.5, 17, 6.5, "CausalConv1d", "32→32", C_TCN, C_TCN)
        arr(bx + 54, y - 3.25, bx + 58, y - 3.25)
        bbox(bx + 58, y - 5.5, 9, 5.0, "⊕", "", "#5a5a5a", "#5a5a5a", fs_t=10)
        arr(bx + 67, y - 3.0, bx + 71, y - 3.0)
        bbox(bx + 71, y - 6.0, 10, 5.5, "ReLU", "", C_ACT, C_ACT, fs_t=7)

        # 残差路径(下方)
        arr(bx + 8.5, y - 6.5, bx + 8.5, y - 12.5, color=C_RES)
        bbox(bx + 1, y - 16.0, 15, 5.5, res_label, "", C_RES, C_RES, fs_t=6.8)
        # 残差并入 ⊕ 底部
        arr(bx + 8.5, y - 13.5, bx + 58, y - 9.5, color=C_RES, ls="--")

        # 块输出标注
        ax.text(bx + 71, y - 10.0, f"→ (32, 16)", fontsize=7.2, color=C_MUTED, ha="center")

    # 块间箭头
    for i in range(3):
        y_top = block_top - i * block_h - 17.5
        y_next = block_top - (i + 1) * block_h + 3.0
        arr(50, y_top, 50, y_next)

    # ---------- Flatten ----------
    last_bottom = block_top - 3 * block_h - 17.5
    arr(50, last_bottom, 50, last_bottom - 4)
    bbox(30, last_bottom - 9, 40, 5.5, "Flatten", "(32 × 16) → 512", C_MUTED, "0.5",
         tc="white")
    arr(50, last_bottom - 9, 50, last_bottom - 13)

    # ---------- 全连接头 ----------
    bbox(30, last_bottom - 18, 40, 5.5, "Linear + ReLU", "512 → 64", C_HEAD, C_HEAD)
    arr(50, last_bottom - 18, 50, last_bottom - 22)
    bbox(30, last_bottom - 27, 40, 5.5, "Linear", "64 → 16", C_HEAD, C_HEAD)
    arr(50, last_bottom - 27, 50, last_bottom - 31)

    # ---------- 输出 ----------
    bbox(30, last_bottom - 36, 40, 5.6, "Output: predicted future",
         "view → (B, 8, 2)", "#f2f2ef", "0.55", tc=C_INK, tc_s=C_INK)

    ax.set_title("TCN trajectory-prediction branch — network architecture "
                 "(4 dilated causal-conv residual blocks + linear head)",
                 fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_tcn_architecture.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)
    print("  +", os.path.join(RESULTS, "fig_tcn_architecture.png"))


def fig_tcn_receptive_field():
    """扩张因果卷积感受野：4 层(d=1,2,4,8,k=3)对最后一个输出可触及的输入时间步。"""
    dilations = [1, 2, 4, 8]
    T = 16
    output = (4, T - 1)

    edges, reachable = [], {output}

    def expand(node):
        l, t = node
        if l == 0:
            return
        d = dilations[l - 1]
        for j in range(3):          # kernel 三个抽头
            tt = t - j * d
            if tt >= 0:
                child = (l - 1, tt)
                edges.append((child, node))
                if child not in reachable:
                    reachable.add(child)
                    expand(child)

    expand(output)
    field_inputs = {t for (l, t) in reachable if l == 0}

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    y_of = lambda l: l + 0.3
    for (child, parent) in edges:
        (cl, ct), (pl, pt) = child, parent
        ax.plot([ct, pt], [y_of(cl), y_of(pl)], color="#9ec5f4", lw=0.8, zorder=1)

    for l in range(5):
        ts = [t for (ll, t) in reachable if ll == l]
        for t in ts:
            if l == 0:
                in_field = t in field_inputs
                ax.scatter(t, y_of(l), s=90, color=C_TCN if in_field else "#d5d5d0",
                           zorder=3, edgecolor="white", linewidth=0.6)
            else:
                ax.scatter(t, y_of(l), s=45, color="#6aa4e8", zorder=3,
                           edgecolor="white", linewidth=0.5)

    # 全部输入时间步(灰色底)
    for t in range(T):
        if (0, t) not in reachable:
            ax.scatter(t, y_of(0), s=90, color="#e8e8e4", zorder=2,
                       edgecolor="white", linewidth=0.6)

    ax.scatter(T - 1, y_of(4), s=120, color=C_ACT, zorder=4, edgecolor="white",
               linewidth=0.8)
    ax.annotate("output", (T - 1, y_of(4)), textcoords="offset points", xytext=(8, 4),
                fontsize=8, color=C_ACT, weight="bold")

    for l, d in enumerate(dilations):
        ax.text(-1.6, y_of(l + 1), f"d={d}", fontsize=8, color=C_MUTED, ha="right",
                va="center")
    ax.text(-1.6, y_of(0), "input", fontsize=8, color=C_MUTED, ha="right", va="center")

    ax.set_xlim(-3.5, T + 0.5); ax.set_ylim(-0.5, 5.4)
    ax.set_xticks(range(T)); ax.set_xlabel("time step (causal, left→right)")
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=C_GRID, lw=0.5)
    ax.set_title("Dilated causal-conv receptive field: 4 layers (d=1,2,4,8, k=3) "
                 "span the full 16-step input", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_tcn_receptive_field.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)
    print("  +", os.path.join(RESULTS, "fig_tcn_receptive_field.png"))


if __name__ == "__main__":
    fig_tcn_architecture()
    fig_tcn_receptive_field()
