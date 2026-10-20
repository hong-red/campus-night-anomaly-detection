#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案1 vs 方案2 对比（全英文 Nature 配图）
========================================

方案1 = FluxEV-SPOT only（EVT 运动分数 + dwell/gather/geofence 规则）
方案2 = FluxEV-SPOT + DL(TCN) 打分分支，max 融合

对四类注入异常（Loitering / Sprinting / Zigzag / Sudden stop），在同一批轨迹上
计算两种方案的逐帧 AUROC / AP，量化 DL 分支带来的增益。

输出：
  results/compare_methods.json           对比结果
  results/figc1_auroc.png/.svg           方案1 vs 方案2 分类型 AUROC
  results/figc2_fusion_time.png/.svg     同一条轨迹上两方案分数时序对比
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tcn_branch as T
from make_figures import BLUE, ORANGE, RED, INK, MUTED, GRID, CHANCE, _save
from sklearn import metrics

RESULTS = T.RESULTS_DIR
KIND = {"loiter": ("Loitering", T.inject_loiter),
        "sprint": ("Sprinting", T.inject_run),
        "zigzag": ("Zigzag", T.inject_zigzag),
        "stop": ("Sudden stop", T.inject_stop)}


def score_one(model, stats, traj, inject_fn, obs, pred):
    """注入异常，返回 (gt, s1_fluxev, s2_fused)。逐帧，长度 n。"""
    x, y, lat, lon, tg, _ = traj
    n = len(x)
    i0 = int(n * 0.55)
    i1 = min(i0 + max(16, int(n * 0.18)), n - obs - pred - 2)
    xa, ya = x.copy(), y.copy()
    inject_fn(xa, ya, i0, i1)
    gt = np.zeros(n); gt[i0:i1] = 1

    s_dl = T.tcn_frame_scores(model, xa, ya, obs, pred, stats, stride=1)
    s_dl = np.nan_to_num(s_dl, nan=0.0)

    lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
    res = T.run_pipeline(lat_a, lon_a, tg, 2.0, others=None, zone=None, poly=None, q=0.005)
    s1 = res["final"]
    s2 = np.maximum(s1, s_dl)
    return gt, s1, s2


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]

    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, 120, seed=7)
    print(f"评估 {len(trajs)} 条轨迹 ...")

    comp = {}
    for key, (en, fn) in KIND.items():
        a1, a2, p1, p2 = [], [], [], []
        for t in trajs:
            n = len(t[0])
            if n < obs + pred + 30:
                continue
            gt, s1, s2 = score_one(model, stats, t, fn, obs, pred)
            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            a1.append(metrics.roc_auc_score(g, s1[mask]))
            a2.append(metrics.roc_auc_score(g, s2[mask]))
            p1.append(metrics.average_precision_score(g, s1[mask]))
            p2.append(metrics.average_precision_score(g, s2[mask]))
        comp[en] = {"auroc_method1": round(float(np.mean(a1)), 3),
                    "auroc_method2": round(float(np.mean(a2)), 3),
                    "ap_method1": round(float(np.mean(p1)), 3),
                    "ap_method2": round(float(np.mean(p2)), 3),
                    "n": len(a1)}
        print(f"  {en:<12} 方案1 AUROC={np.mean(a1):.3f}  ->  方案2 AUROC={np.mean(a2):.3f}  "
              f"(AP {np.mean(p1):.3f} -> {np.mean(p2):.3f})")

    json.dump(comp, open(os.path.join(RESULTS, "compare_methods.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- figc1: 分组条形图 ----
    names = list(comp.keys())
    m1 = [comp[k]["auroc_method1"] for k in names]
    m2 = [comp[k]["auroc_method2"] for k in names]
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    xs = np.arange(len(names))
    w = 0.36
    b1 = ax.bar(xs - w / 2, m1, w, color="#9ec5f4", label="Method 1: FluxEV only", zorder=3)
    b2 = ax.bar(xs + w / 2, m2, w, color=BLUE, label="Method 2: FluxEV + TCN", zorder=3)
    ax.axhline(0.5, color=CHANCE, lw=0.9, ls="--", zorder=2)
    for x, v in zip(xs - w / 2, m1):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5, color=MUTED)
    for x, v in zip(xs + w / 2, m2):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    _save(fig, "figc1_auroc")

    # ---- figc2: 时序对比（一条轨迹，徘徊） ----
    traj = max(trajs, key=lambda t: len(t[0]))
    gt, s1, s2 = score_one(model, stats, traj, T.inject_loiter, obs, pred)
    n = len(s1)
    tt = traj[4]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.fill_between(tt, 0, 1, where=gt > 0, color=RED, alpha=0.10, lw=0, label="Anomaly (GT)")
    ax.plot(tt, s1, color="#9ec5f4", lw=1.3, label="Method 1: FluxEV only")
    ax.plot(tt, s2, color=BLUE, lw=1.5, label="Method 2: FluxEV + TCN (fused)")
    ax.axhline(0.5, color=CHANCE, lw=0.9, ls="--")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Anomaly score")
    ax.set_ylim(0, 1.02)
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "figc2_fusion_time")

    print("\n完成。compare_methods.json / figc1 / figc2 已输出到 results/")


if __name__ == "__main__":
    main()
