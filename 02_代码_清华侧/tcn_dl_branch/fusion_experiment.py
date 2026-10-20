#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相融实验：FluxEV × TCN 的融合策略对比
=====================================

背景：方案2 用 naive `max(S_ev, S_dl)` 融合，在运动异常(疾跑/徘徊/蛇形)上大幅增益，
但把 TCN 的散点噪声带进"急停"，AUROC 从 0.618 掉到 0.368。本脚本在同一批轨迹上
系统扫几类融合策略，回答"怎么融才不吃掉规则域的收益"：

  1. max                 —— 基线（当前方案2）
  2. S_dl 置信门控        —— S_dl 必须超过阈值 θ 才参与（抑制散点噪声）
  3. 时序平滑 smooth      —— 对 S_dl 做因果滑动平均，压掉孤立尖峰
  4. 速度门控 speed       —— 只有人在走(瞬时速度≥v)才信 TCN（静止交给 FluxEV dwell）
  5. 逗留门控 dwell       —— FluxEV dwell 已报高的时段不再叠加 TCN（互补路由）
  6. 组合策略

输出：
  results/fusion_experiment.json    全量 AUROC/AP
  results/fusion_experiment.png     S_dl 门控阈值 vs 四类异常 AUROC 的折中曲线
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
from sklearn import metrics

import tcn_branch as T

RESULTS = T.RESULTS_DIR
KIND = {"loiter": ("Loitering", T.inject_loiter),
        "sprint": ("Sprinting", T.inject_run),
        "zigzag": ("Zigzag", T.inject_zigzag),
        "stop": ("Sudden stop", T.inject_stop)}

# 融合变体：(名称, {smooth_w, dl_gate, speed_gate, dwell_gate})
VARIANTS = [
    ("max (baseline)",       dict(smooth_w=1, dl_gate=0.0, speed_gate=0.0, dwell_gate=0.0)),
    ("S_dl>=0.5",            dict(smooth_w=1, dl_gate=0.5, speed_gate=0.0, dwell_gate=0.0)),
    ("S_dl>=0.7",            dict(smooth_w=1, dl_gate=0.7, speed_gate=0.0, dwell_gate=0.0)),
    ("S_dl>=0.85",           dict(smooth_w=1, dl_gate=0.85, speed_gate=0.0, dwell_gate=0.0)),
    ("smooth5",              dict(smooth_w=5, dl_gate=0.0, speed_gate=0.0, dwell_gate=0.0)),
    ("smooth9",              dict(smooth_w=9, dl_gate=0.0, speed_gate=0.0, dwell_gate=0.0)),
    ("smooth5+S_dl>=0.7",    dict(smooth_w=5, dl_gate=0.7, speed_gate=0.0, dwell_gate=0.0)),
    ("speed>=0.5",           dict(smooth_w=1, dl_gate=0.0, speed_gate=0.5, dwell_gate=0.0)),
    ("dwell-gate",           dict(smooth_w=1, dl_gate=0.0, speed_gate=0.0, dwell_gate=0.5)),
    ("spd0.5+S_dl>=0.7",     dict(smooth_w=1, dl_gate=0.7, speed_gate=0.5, dwell_gate=0.0)),
    ("spd0.5+s5+S_dl>=0.7",  dict(smooth_w=5, dl_gate=0.7, speed_gate=0.5, dwell_gate=0.0)),
    ("dwell-gate+S_dl>=0.7", dict(smooth_w=1, dl_gate=0.7, speed_gate=0.0, dwell_gate=0.5)),
]


def causal_roll(x, w):
    """因果滑动平均（只看过去），w<=1 时原样返回。"""
    x = np.asarray(x, dtype=np.float64)
    if w <= 1:
        return x
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x)])
    tail = (c[w:] - c[:-w]) / w
    head = np.cumsum(x[:w - 1]) / np.arange(1, w)
    return np.concatenate([head, tail])[:n]


def fusion(s_ev, s_dl, speed, dwell, **v):
    """按变体参数把 FluxEV 与 TCN 分数融合。"""
    sd = np.nan_to_num(s_dl, nan=0.0)
    if v.get("smooth_w", 1) > 1:
        sd = causal_roll(sd, v["smooth_w"])
    contrib = sd.copy()
    if v.get("dl_gate", 0.0) > 0:
        contrib = np.where(sd >= v["dl_gate"], contrib, 0.0)
    if v.get("speed_gate", 0.0) > 0:
        contrib = np.where(speed >= v["speed_gate"], contrib, 0.0)
    if v.get("dwell_gate", 0.0) > 0:
        contrib = np.where(dwell < v["dwell_gate"], contrib, 0.0)
    return np.maximum(s_ev, contrib)


def score_one(model, stats, traj, inject_fn, obs, pred):
    """注入异常，返回逐帧 (gt, s_ev, s_dl, speed, dwell)。"""
    x, y, lat, lon, tg, _ = traj
    n = len(x)
    i0 = int(n * 0.55)
    i1 = min(i0 + max(16, int(n * 0.18)), n - obs - pred - 2)
    xa, ya = x.copy(), y.copy()
    inject_fn(xa, ya, i0, i1)
    gt = np.zeros(n); gt[i0:i1] = 1

    s_dl = T.tcn_frame_scores(model, xa, ya, obs, pred, stats, stride=1)

    lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
    res = T.run_pipeline(lat_a, lon_a, tg, 2.0, others=None, zone=None, poly=None, q=0.005)
    return gt, res["final"], s_dl, res["speed"], res["dwell"]


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, 120, seed=7)
    print(f"评估 {len(trajs)} 条轨迹 x {len(KIND)} 类异常 x {len(VARIANTS)} 种融合 ...")

    # 汇总容器：variants[(en, vname)] = [auroc...];  sweep[(en, smooth_w, gate)] = [auroc...]
    var_auroc = {(en, vname): [] for en, _ in KIND.values() for vname, _ in VARIANTS}
    var_ap = {(en, vname): [] for en, _ in KIND.values() for vname, _ in VARIANTS}
    gates = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    smooths = [1, 5]
    sweep_auroc = {(en, sw, g): [] for en, _ in KIND.values() for sw in smooths for g in gates}

    for key, (en, fn) in KIND.items():
        for t in trajs:
            n = len(t[0])
            if n < obs + pred + 30:
                continue
            gt, s_ev, s_dl, speed, dwell = score_one(model, stats, t, fn, obs, pred)
            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            sev, sd, sp, dw = s_ev[mask], s_dl[mask], speed[mask], dwell[mask]
            # 变体表
            for vname, v in VARIANTS:
                f = fusion(sev, sd, sp, dw, **v)
                var_auroc[(en, vname)].append(metrics.roc_auc_score(g, f))
                var_ap[(en, vname)].append(metrics.average_precision_score(g, f))
            # 门控阈值扫描（平滑窗 ∈ {1,5}）
            for sw in smooths:
                base = causal_roll(np.nan_to_num(sd, nan=0.0), sw)
                for gt_ in gates:
                    f = np.maximum(sev, np.where(base >= gt_, base, 0.0))
                    sweep_auroc[(en, sw, gt_)].append(metrics.roc_auc_score(g, f))

    # ---- 汇总打印 ----
    print("\n" + "=" * 100)
    print("融合策略 × 四类异常 AUROC（逐轨迹均值，测试段 = 后半段）")
    print("=" * 100)
    header = f"{'fusion strategy':<22}" + "".join(f"{en:<13}" for en, _ in KIND.values())
    print(header)
    print("-" * 100)
    out = {}
    for vname, v in VARIANTS:
        row = f"{vname:<22}"
        out[vname] = {}
        for en, _ in KIND.values():
            a = np.mean(var_auroc[(en, vname)])
            out[vname][en] = {"auroc": round(float(a), 3),
                              "ap": round(float(np.mean(var_ap[(en, vname)])), 3)}
            row += f"{a:<13.3f}"
        print(row)

    # 找出"最稳健"融合：四类 AUROC 最小值最大者（naive max 的 min 只有 0.368）
    best = max(VARIANTS, key=lambda vn: min(np.mean(var_auroc[(en, vn[0])]) for en, _ in KIND.values()))
    print("-" * 100)
    print(f"最稳健融合（四类最小 AUROC 最大）: {best[0]}")
    for en, _ in KIND.values():
        print(f"    {en:<12} AUROC={np.mean(var_auroc[(en, best[0])]):.3f}")

    json.dump(out, open(os.path.join(RESULTS, "fusion_experiment.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- 门控阈值折中曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    colors = {"Loitering": "#0B6BFF", "Sprinting": "#E8272C",
              "Zigzag": "#00B140", "Sudden stop": "#FF7A00"}
    for sw, ax in zip(smooths, axes):
        for en, _ in KIND.values():
            y = [np.mean(sweep_auroc[(en, sw, g)]) for g in gates]
            ax.plot(gates, y, marker="o", ms=4, lw=1.4, color=colors[en], label=en)
        ax.axhline(0.5, color="#999999", lw=0.8, ls="--")
        ax.set_title(f"S_dl gate threshold sweep (smooth={sw})")
        ax.set_xlabel("S_dl confidence gate θ"); ax.set_ylabel("AUROC")
        ax.set_ylim(0.3, 0.9)
        ax.grid(True, color="#eeeeee", lw=0.5)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fusion_experiment.png"), dpi=150)
    plt.close(fig)
    print("\n完成。fusion_experiment.json / fusion_experiment.png 已输出到 results/")


if __name__ == "__main__":
    main()
