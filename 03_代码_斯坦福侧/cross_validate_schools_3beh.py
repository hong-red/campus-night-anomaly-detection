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
"""两校互为验证集（跨数据集泛化）—— 只跑 3 个运动行为：徘徊/疾跑/之字形。

复用已训练的 checkpoint（geo_tcn_dt1.pt / sdd_tcn_checkpoint.pt），不重训；
只做测试阶段 window 级 AUROC（dt=1s，同 cross_validate_schools.py 口径）。

组合：Geo→Geo(清华域内) / Geo→SDD(迁移) / SDD→SDD(斯坦福域内) / SDD→Geo(迁移)
输出：results/cross_validate_schools_3beh.json + .png
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, _PKGP.CODE_TSINGHUA, _PKGP.TCN_DIR,
           _PKGP.STYLE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sdd_common as S
import tcn_branch as T
import cross_validate_schools as CS
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

GEO_TCN_DT1 = os.path.join(S.RESULTS_DIR, "geo_tcn_dt1.pt")
BEHAVIORS = [("wandering", "Wandering", T.inject_loiter),
             ("sprint", "Sprint", T.inject_run),
             ("zigzag", "Zigzag", T.inject_zigzag)]


def main():
    combos = [("geo_geo", "Geo→Geo (in)"),
              ("geo_sdd", "Geo→SDD (trans)"),
              ("sdd_sdd", "SDD→SDD (in)"),
              ("sdd_geo", "SDD→Geo (trans)")]

    print("读取现成帧级数据（cross_validate_schools_frame.json）...")
    _frame_json = os.path.join(S.RESULTS_DIR, "cross_validate_schools_frame.json")
    frame = json.load(open(_frame_json, encoding="utf-8"))
    print(f"  {_frame_json}")

    print("\n帧级跨校 AUROC（3 行为 × 4 train→test 组合）：")
    matrix = {k: {ck: frame[k][ck] for ck, *_ in combos} for k, _, _ in BEHAVIORS}

    hdr = "".join(f"{cname:<22}" for _, cname in combos)
    print(f"{'behavior':<12}{hdr}")
    print("-" * 96)
    for key, en, _ in BEHAVIORS:
        print(f"{en:<14}" + "".join(f"{matrix[key][ck]:<22.3f}" for ck, *_ in combos))

    # 迁移均值 vs 域内均值
    print("-" * 96)
    for ck, cname, *_ in combos:
        vals = [matrix[k][ck] for k, _, _ in BEHAVIORS]
        print(f"  {cname:<22} mean AUROC = {np.mean(vals):.3f}")

    json.dump(matrix, open(os.path.join(S.RESULTS_DIR, "cross_validate_schools_3beh.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- 图（横向条形图） ----
    COLOR = {"geo_geo": "#9ec5f4", "geo_sdd": "#2a78d6",
             "sdd_sdd": "#f6c9a8", "sdd_geo": "#eb6834"}
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    y = np.arange(len(BEHAVIORS)); h = 0.19
    for j, (ck, cname, *_rest) in enumerate(combos):
        off = -1.5 * h + j * h
        vals = [matrix[k][ck] for k, _, _ in BEHAVIORS]
        ax.barh(y + off, vals, h, color=COLOR[ck], zorder=3, label=cname)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.012, yi, f"{v:.2f}", ha="left", va="center", fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels([en for _, en, _ in BEHAVIORS], fontsize=18)
    ax.set_xlabel("AUROC", fontsize=22)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              fontsize=17, handlelength=1.2, columnspacing=1.6)
    ax.set_title("Frame-level cross-dataset generalization (AUROC)",
                 fontsize=25, pad=34)
    
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "cross_validate_schools_3beh.png", outdir=S.RESULTS_DIR)
    print("\nsaved results/cross_validate_schools_3beh.json + .png")


if __name__ == "__main__":
    main()
