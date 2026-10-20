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
    obs, pred = 16, 8

    print("加载 checkpoint（不重训）...")
    geo_model, _, _ = T.load_predictor(GEO_TCN_DT1)
    sdd_model, _, _ = S.ensure_sdd_tcn()
    print(f"  Geo 模型: {GEO_TCN_DT1}")
    print(f"  SDD 模型: {S.SDD_TCN_CKPT}")

    print("准备测试集(dt=1s, 留出, seed=42)...")
    sdd_test = [(x, y) for _, x, y, _ in S.collect_sdd_walking(1.0, 100, seed=42, min_len=48)]
    geo_test = [(x, y) for x, y, *_ in T.collect_walking_trajectories(T.DEFAULT_DATA, 1.0,
                                                                      100, seed=42)]
    print(f"  Stanford 测试轨迹 = {len(sdd_test)}，Geolife 测试轨迹 = {len(geo_test)}")

    combos = [("geo_geo", "Geo→Geo (in)", geo_model, geo_test),
              ("geo_sdd", "Geo→SDD (trans)", geo_model, sdd_test),
              ("sdd_sdd", "SDD→SDD (in)", sdd_model, sdd_test),
              ("sdd_geo", "SDD→Geo (trans)", sdd_model, geo_test)]

    print("\n交叉验证 AUROC（3 行为 × 4 train→test 组合）：")
    matrix = {k: {} for k, _, _ in BEHAVIORS}
    for key, en, inj in BEHAVIORS:
        for ck, cname, model, tracks in combos:
            matrix[key][ck] = round(CS.window_auroc(model, obs, pred, tracks, inj), 3)

    hdr = "".join(f"{cname:<22}" for _, cname, _, _ in combos)
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

    # ---- 图 ----
    COLOR = {"geo_geo": "#9ec5f4", "geo_sdd": "#2a78d6",
             "sdd_sdd": "#f6c9a8", "sdd_geo": "#eb6834"}
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(BEHAVIORS)); w = 0.19
    for j, (ck, cname, *_rest) in enumerate(combos):
        off = -1.5 * w + j * w
        vals = [matrix[k][ck] for k, _, _ in BEHAVIORS]
        ax.bar(x + off, vals, w, color=COLOR[ck], zorder=3, label=cname)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([en for _, en, _ in BEHAVIORS], fontsize=18)
    ax.set_ylabel("AUROC", fontsize=22)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=17,
              handlelength=1.2, columnspacing=1.0)
    ax.set_title("Cross-dataset generalization — TCN branch (3 motion behaviours)",
                 fontsize=25, pad=10)
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "cross_validate_schools_3beh.png", outdir=S.RESULTS_DIR)
    print("\nsaved results/cross_validate_schools_3beh.json + .png")


if __name__ == "__main__":
    main()
