#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小绘图风格桩（make_figures.py 原出图脚本已在归档时移除，仅保留 compare_methods.py
等仍引用其颜色常量与 _save 的接口）。颜色值与同目录 eval_dwell_wander.py / style.py
的 Nature 风配色一致，仅影响出图观感，不影响任何数据结果。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
CHANCE = "#c3c2b7"

_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _save(fig, name, outdir=None):
    """保存 figure 为 results/<name>.png + .svg（300 dpi），并关闭图。"""
    d = outdir or _RESULTS
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.abspath(os.path.join(d, name)))
