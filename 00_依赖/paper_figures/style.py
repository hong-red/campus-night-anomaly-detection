#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校园夜间安防项目 —— 论文绘图统一风格（Nature 投稿风格）。

把散落在各 make_*.py 里的配色/字号/尺寸/去轴约定收敛到这一个文件，
以后所有图都 `from style import ...`，保证整篇论文图风一致。

用法
----
    import sys, os; sys.path.insert(0, os.path.dirname(__file__))   # style.py 所在目录
    import matplotlib.pyplot as plt
    from style import (apply_nature_style, apply_cn_style, C, BRANCH_COLOR,
                       DOMAIN_COLOR, SINGLE_COL, DOUBLE_COL, MM,
                       panel_label, save, despine)
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 48 * MM))
    ...
    save(fig, "fig_x.png", outdir="results/figs")
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 尺寸 ----------------
MM = 1 / 25.4
SINGLE_COL = 89 * MM      # 单栏宽（Nature 单栏 89mm）
DOUBLE_COL = 183 * MM     # 双栏宽（183mm）

# ---------------- 配色 ----------------
# 基础低饱和配色（Nature 风，来自 make_nature_figs.py）
C = {
    "alert":  "#B2182B",   # 异常 / 告警         红
    "calm":   "#4393C3",   # 正常 / 运动 / FluxEV 蓝
    "accent": "#D6604D",   # 强调 / 徘徊          锈橙
    "grey":   "#6E6E6E",   # 灰（参考线 / 杂项）
    "dark":   "#252525",   # 主曲线               近黑
    "green":  "#1B7837",   # 规则 / 围栏          绿
}

# 分支语义色（三路消融 / 融合表，来自 fusion_ablation.py）
BRANCH_COLOR = {
    "fluxev": "#1B7837",   # 硬 FluxEV（EVT 运动 + 规则）绿
    "dl":     "#eb6834",   # 软 TCN（深度学习）        橙
    "fused":  "#0b0b0b",   # 组合 硬+软               黑
    "rules":  "#9AA0A6",   # 仅 if 规则               灰
}

# 域语义色（规则域 vs 运动域）
DOMAIN_COLOR = {
    "rule": "#1B7837",     # 规则域（逗留/闯入/聚集）绿
    "tcn":  "#eb6834",     # TCN 运动域（徘徊/疾跑/蛇形）橙
    "evt":  "#4393C3",     # EVT 运动（急停）          蓝
}


# ---------------- 样式 ----------------

def apply_nature_style():
    """Nature 投稿风格：Arial、350 dpi、去顶/右脊线、面板字母、低饱和。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "lines.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 350,
        "savefig.dpi": 350,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.unicode_minus": False,
    })


def apply_cn_style():
    """中文标注风格（报告 / 答辩图）：SimHei + 米灰底 + 去轴，非投稿版。"""
    from matplotlib import font_manager as fm
    for _fp in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(_fp):
            fm.fontManager.addfont(_fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
            break
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
    })


# ---------------- 辅助 ----------------

def panel_label(ax, s, dx=-0.10, dy=1.06):
    """面板字母 a/b/c（Nature 惯例，右上角粗体）。"""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="right")


def despine(ax, color="#cccccc"):
    """去掉顶/右脊线，底/左脊线淡灰（适用于非 rcParams 全局去轴时）。"""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(color)


def save(fig, name, outdir="results/figs"):
    """保存 + 关图，返回保存路径。outdir 自动创建。"""
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p)
    plt.close(fig)
    print("saved", os.path.abspath(p))
    return p


def chance_line(ax, x=0.5, color="#6E6E6E", orient="h"):
    """随机基线 0.5 虚线（orient: 'h' 横轴 / 'v' 纵轴）。"""
    if orient == "h":
        ax.axhline(x, ls=(0, (4, 3)), color=color, lw=0.6, zorder=2)
    else:
        ax.axvline(x, ls=(0, (4, 3)), color=color, lw=0.6, zorder=2)


# ---------------- 架构 / 流程图（模块块风格，参考手写笔记纵向流程） ----------------

DIAGRAM_COLOR = {
    "input":  {"fill": "#DDE9F5", "edge": "#6E9EC4"},   # 输入 / 数据 / 指标 蓝
    "frozen": {"fill": "#E8E8E8", "edge": "#9A9A9A"},   # 无训练（FluxEV / 规则）灰
    "train":  {"fill": "#FCE5D0", "edge": "#D08A3E"},   # 可训练（TCN）橙
    "fusion": {"fill": "#DCEAD7", "edge": "#7A9C7A"},   # 融合 / 输出 绿
}
DIAGRAM_ARROW = "#555555"


def diagram_box(ax, cx, cy, w, h, lines, c="frozen", fsz=11,
                ink="#333333", bold_first=True):
    """模块块：圆角矩形 + 多行文本，第 0 行加粗当标题。
    c 取 DIAGRAM_COLOR 的键（input/frozen/train/fusion）或直接传 {fill, edge}。"""
    from matplotlib.patches import FancyBboxPatch
    cc = DIAGRAM_COLOR[c] if isinstance(c, str) else c
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.15",
                 fc=cc["fill"], ec=cc["edge"], lw=1.3, zorder=3))
    n = len(lines)
    for i, s in enumerate(lines):
        yy = cy + (n - 1 - i) * (h * 0.30) - (n - 1) * (h * 0.30) / 2
        ax.text(cx, yy, s, ha="center", va="center", fontsize=fsz, color=ink,
                fontweight=("bold" if (i == 0 and bold_first) else "normal"),
                zorder=4)


def diagram_arrow(ax, x1, y1, x2, y2, rad=None, lw=1.5, color=DIAGRAM_ARROW):
    """模块间箭头；rad 给弧线弯曲（如 0.2 / -0.2 分叉）。"""
    from matplotlib.patches import FancyArrowPatch
    kw = dict(arrowstyle="-|>", mutation_scale=16, lw=lw, color=color, zorder=2)
    if rad is not None:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))
