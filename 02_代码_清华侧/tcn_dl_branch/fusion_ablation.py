#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三路消融：硬性 if 规则 vs 深度学习(TCN) vs 相加融合 —— 支撑论文「表 1」
=======================================================================

问题
----
单独用「硬性 if 条件」或单独用「深度学习」各只能覆盖一类事件，把它们相加(融合)后
才能同时覆盖全部事件。本实验在真实 Geolife 行走轨迹上，对 7 类事件做统一消融，
用 AUROC 证明「软性条件 + 硬性条件相加效果更好」。

事件分类（表 1）
----------------
  规则域事件（规则分支负责，静止/空间/共现对 TCN 不可见）:
     dwell 长时间停留 / geofence 地理围栏突破 / gather 非工作时间聚集
  运动模式事件（深度学习分支负责）:
     wandering 徘徊 / sprint 冲刺 / zigzag / sudden-stop 突然停止

四个检测器
------------
  rules   硬性 if 规则 = max(dwell, gather, geofence)
  dl      深度学习 = z-clip(逐帧 ADE)              （真零地板，正常→0，即置信度门限）
  fused   相加 = max(rules, dl)                     （软 + 硬，本实验核心）
  full    完整系统 = max(FluxEV, dl)                （含 EVT 运动，用于急停）

输出 (tcn_dl_branch/results/)
-----------------------------
  fusion_ablation.json          7 事件 × 4 检测器 AUROC 矩阵
  fusion_ablation_bars.png      分组条形图（AUROC，0.5 随机线）
  fusion_ablation_table.png     表 1（事件 → 负责分支）
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# Windows 下必须先 import torch 再 import numpy（c10.dll 冲突）
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import metrics
import tcn_branch as T
import geolife_pipeline as gp

RESULTS = T.RESULTS_DIR
CKPT = T.CKPT_PATH

DETECTORS = ["fluxev", "dl", "fused", "rules"]
DET_EN = {"fluxev": "Hard (FluxEV: EVT+rules)",
          "dl": "Soft (TCN)",
          "fused": "Fused (hard + soft)",
          "rules": "if-rules only (subset)"}
DET_COLOR = {"fluxev": "#1B7837", "dl": "#eb6834",
             "fused": "#0b0b0b", "rules": "#9AA0A6"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9.0,
    "legend.fontsize": 8.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "savefig.dpi": 220,
})


# --------------------------- 分数工具 ---------------------------

def frame_ade(model, x, y, obs_len, pred_len):
    """整条轨迹逐帧原始 ADE(米)，无窗口处为 NaN（批量计算）。"""
    n = len(x)
    total = obs_len + pred_len
    if n < total:
        return np.full(n, np.nan)
    obs_list, fut_list = [], []
    for s in range(0, n - total + 1):
        ref = np.array([x[s + obs_len - 1], y[s + obs_len - 1]])
        obs_list.append(np.stack([x[s:s + obs_len], y[s:s + obs_len]], -1) - ref)
        fut_list.append(np.stack([x[s + obs_len:s + total], y[s + obs_len:s + total]], -1) - ref)
    ade_all = T.compute_ade(model, np.asarray(obs_list, np.float32),
                            np.asarray(fut_list, np.float32))
    ade = np.full(n, np.nan)
    ade[obs_len:obs_len + len(ade_all)] = ade_all
    return ade


def zclip(ade, mu, sigma, k=3.0):
    """ADE → [0,1] z-clip 标定（真零地板，等价于置信度门限）。"""
    ade = np.nan_to_num(ade, nan=mu)
    return np.clip((ade - mu) / (k * sigma + 1e-8), 0.0, 1.0)


def _auroc(gt, score):
    if (gt == 1).any() and (gt == 0).any() and np.std(score) > 1e-9:
        return float(metrics.roc_auc_score(gt, score))
    return 0.5


def four_scores(gt, rules, dl, fluxev):
    """由 rules / dl / fluxev 组装四个检测器，返回 {det: score}。

    fluxev = 硬性分支(FluxEV = EVT运动 + 规则)   dl = 软性(TCN)
    fused = max(fluxev, dl)  = 硬 + 软 相加       rules = 仅 if 规则(硬性里的子集)
    """
    return {"fluxev": fluxev,
            "dl": dl,
            "fused": np.maximum(fluxev, dl),   # 相加 = 硬性 + 软性
            "rules": rules}                     # if 规则子集（供参考）


# --------------------------- 运动模式事件（dt=2s，多轨迹） ---------------------------

MOTION = [("wandering", T.inject_loiter), ("sprint", T.inject_run),
          ("zigzag", T.inject_zigzag), ("sudden-stop", T.inject_stop)]


def score_motion(model, mu, sigma, traj, inject_fn, obs, pred):
    """注入运动异常，返回 (gt, {det: score})。"""
    x, y, lat, lon, tg, _ = traj
    n = len(x)
    i0 = int(n * 0.55)
    i1 = min(i0 + max(16, int(n * 0.18)), n - obs - pred - 2)   # 0.18n ≈ 1~2.5min
    xa, ya = x.copy(), y.copy()
    inject_fn(xa, ya, i0, i1)
    gt = np.zeros(n); gt[i0:i1] = 1

    lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
    res = T.run_pipeline(lat_a, lon_a, tg, 2.0, others=None, zone=None, poly=None, q=0.005)
    rules = np.maximum.reduce([res["dwell"], res["gather"], res["geofence"]])
    dl = zclip(frame_ade(model, xa, ya, obs, pred), mu, sigma)
    return gt, four_scores(gt, rules, dl, res["final"])


def run_motion_benchmark(model, stats, trajs, obs, pred):
    """运动事件：逐轨迹 AUROC 求均值，返回 {event: {det: auroc}}。"""
    mu, sigma = stats["mu"], stats["sigma"]
    agg = {(en, d): [] for en, _ in MOTION for d in DETECTORS}
    for en, fn in MOTION:
        for t in trajs:
            if len(t[0]) < obs + pred + 30:
                continue
            gt, sc = score_motion(model, mu, sigma, t, fn, obs, pred)
            mask = np.arange(len(gt)) >= len(gt) // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            for d in DETECTORS:
                agg[(en, d)].append(_auroc(g, sc[d][mask]))
    return {en: {d: round(float(np.mean(agg[(en, d)])), 3) for d in DETECTORS}
            for en, _ in MOTION}


# --------------------------- 规则域事件（dt=2s，单条超长轨迹） ---------------------------

def build_rule_scenario(lat, lon, t, dt):
    """在超长行走轨迹上注入 dwell / gather / geofence 三类规则域事件。"""
    n = len(lat)
    lat = lat.copy(); lon = lon.copy()
    gt = {k: np.zeros(n, int) for k in ("dwell", "gather", "geofence")}
    test_start = n // 2

    # 1) dwell：360s 原地停留
    d0 = test_start + int(150 / dt); d1 = d0 + int(360 / dt)
    lat[d0:d1] = lat[d0]; lon[d0:d1] = lon[d0]
    gt["dwell"][d0:d1] = 1

    # 2) gather：两个"他人"在 12m 内共现 300s（半径 50m 内，规则应触发）
    g0 = test_start + int(1200 / dt); g1 = g0 + int(300 / dt)
    kx = 111320.0 * np.cos(np.radians(lat[g0])); ky = 110540.0
    others = []
    for dx, dy in ((12.0, 0.0), (0.0, 12.0)):
        o_lat = np.full(n, lat[g0] + 200.0 / ky)     # 汇聚前远离 200m
        o_lon = np.full(n, lon[g0] + 200.0 / kx)
        o_lat[g0:g1] = lat[g0:g1] + dy / ky          # 跟随主行人，保持 12m 共现
        o_lon[g0:g1] = lon[g0:g1] + dx / kx
        others.append((o_lat, o_lon))
    gt["gather"][g0:g1] = 1

    # 3) geofence：轨迹穿过 60m 圆
    c = test_start + int(2100 / dt)
    zone = (float(lat[c]), float(lon[c]), 60.0)
    gt["geofence"] = (gp.geofence_score(lat, lon, zone) > 0).astype(int)

    return lat, lon, t, gt, others, zone


def run_rule_benchmark(model, stats, obs, pred):
    """规则事件：超长轨迹注入，逐事件 AUROC。"""
    mu, sigma = stats["mu"], stats["sigma"]
    walks = gp.scan_walking_trajectories(T.DEFAULT_DATA, max_files=2000, min_dur=900)
    if not walks:
        raise RuntimeError("未找到超长行走轨迹")
    npts, med, path, lat5, lon5, tg5 = walks[0]
    print(f"  规则域基准轨迹: {os.path.basename(path)}  (dt=5s 点数={npts}, "
          f"时长={tg5[-1]/60:.0f}min)")

    lat, lon, tg = gp.resample(lat5, lon5, tg5, 2.0)
    lat, lon, t, gt, others, zone = build_rule_scenario(lat, lon, tg, 2.0)

    res = gp.run_pipeline(lat, lon, t, 2.0, others=others, zone=zone, q=0.005)
    rules = np.maximum.reduce([res["dwell"], res["gather"], res["geofence"]])
    x, y = T.latlon_to_local(lat, lon)
    dl = zclip(frame_ade(model, x, y, obs, pred), mu, sigma)
    sc = four_scores(gt, rules, dl, res["final"])

    test_mask = np.arange(len(lat)) >= len(lat) // 2
    out = {}
    for kind in ("dwell", "gather", "geofence"):
        g = gt[kind][test_mask]
        out[kind] = {d: round(_auroc(g, sc[d][test_mask]), 3) for d in DETECTORS}
    return out


# --------------------------- 绘图 ---------------------------

def fig_bars(matrix):
    order = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag", "sudden-stop"]
    events = [e for e in order if e in matrix]
    en_label = {"dwell": "Dwell\n(loiter)", "geofence": "Geofence\n(intrusion)",
                "gather": "Gather\n(>=3 ppl)", "wandering": "Wandering",
                "sprint": "Sprint", "zigzag": "Zigzag", "sudden-stop": "Sudden\nstop"}

    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    x = np.arange(len(events))
    w = 0.19
    for j, d in enumerate(DETECTORS):
        off = -1.5 * w + j * w
        vals = [matrix[e][d] for e in events]
        ax.bar(x + off, vals, w, color=DET_COLOR[d], zorder=3, label=DET_EN[d])
    ax.axhline(0.5, color="#c3c2b7", lw=1.0, ls="--", zorder=2)
    ax.text(len(events) - 0.5, 0.505, "chance 0.5", ha="right", va="bottom",
            fontsize=7.5, color="#898781")
    ax.set_xticks(x)
    ax.set_xticklabels([en_label[e] for e in events], fontsize=7.5)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Ablation across 7 real-trajectory events: hard FluxEV / soft TCN / fused",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="y", color="#e1e0d9", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fusion_ablation_bars.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_table(matrix):
    rows = [
        ["Event", "Responsible branch", "Hard (FluxEV)", "Soft (TCN)", "Fused", "if-rules"],
        ["Dwell (>5 min)", "Hard (FluxEV rules)", f"{matrix['dwell']['fluxev']:.2f}",
         f"{matrix['dwell']['dl']:.2f}", f"{matrix['dwell']['fused']:.2f}", f"{matrix['dwell']['rules']:.2f}"],
        ["Geofence intrusion", "Hard (FluxEV rules)", f"{matrix['geofence']['fluxev']:.2f}",
         f"{matrix['geofence']['dl']:.2f}", f"{matrix['geofence']['fused']:.2f}", f"{matrix['geofence']['rules']:.2f}"],
        ["Off-hours gathering", "Hard (FluxEV rules)", f"{matrix['gather']['fluxev']:.2f}",
         f"{matrix['gather']['dl']:.2f}", f"{matrix['gather']['fused']:.2f}", f"{matrix['gather']['rules']:.2f}"],
        ["Wandering", "Soft (TCN)", f"{matrix['wandering']['fluxev']:.2f}",
         f"{matrix['wandering']['dl']:.2f}", f"{matrix['wandering']['fused']:.2f}", f"{matrix['wandering']['rules']:.2f}"],
        ["Sprint", "Soft (TCN)", f"{matrix['sprint']['fluxev']:.2f}",
         f"{matrix['sprint']['dl']:.2f}", f"{matrix['sprint']['fused']:.2f}", f"{matrix['sprint']['rules']:.2f}"],
        ["Zigzag", "Soft (TCN)", f"{matrix['zigzag']['fluxev']:.2f}",
         f"{matrix['zigzag']['dl']:.2f}", f"{matrix['zigzag']['fused']:.2f}", f"{matrix['zigzag']['rules']:.2f}"],
        ["Sudden stop", "Hard (EVT motion) + gate", f"{matrix['sudden-stop']['fluxev']:.2f}",
         f"{matrix['sudden-stop']['dl']:.2f}", f"{matrix['sudden-stop']['fused']:.2f}", f"{matrix['sudden-stop']['rules']:.2f}"],
    ]
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.axis("off")
    tab = ax.table(cellText=rows, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.2)
    tab.scale(1, 1.6)
    for j in range(len(rows[0])):
        tab[(0, j)].set_text_props(fontweight="bold")
    fig.suptitle("Table 1 — event taxonomy, responsible branch, and AUROC",
                 fontsize=10, y=0.995)
    fig.text(0.5, 0.01,
             "Note: naive max(S_ev, S_dl) drops sudden-stop AUROC 0.62 -> 0.37; a confidence gate on S_dl (z-clip, zero floor) recovers it.",
             ha="center", va="bottom", fontsize=7.5, color="#898781")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(RESULTS, "fusion_ablation_table.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# --------------------------- 主流程 ---------------------------

def main():
    model, stats, cfg = T.load_predictor(CKPT)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    print(f"模型: {CKPT}  (obs={obs}, pred={pred}, mu={mu:.2f}m, sigma={sigma:.2f}m)")

    print("\n[1/2] 运动模式事件基准 (dt=2s) ...")
    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, 120, seed=7)
    print(f"  评估轨迹数 = {len(trajs)}")
    matrix = run_motion_benchmark(model, stats, trajs, obs, pred)

    print("[2/2] 规则域事件基准 (dt=2s, 超长轨迹) ...")
    matrix.update(run_rule_benchmark(model, stats, obs, pred))

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(matrix, open(os.path.join(RESULTS, "fusion_ablation.json"), "w"),
              ensure_ascii=False, indent=2)

    order = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag", "sudden-stop"]
    print("\n" + "=" * 78)
    print("AUROC 矩阵  (7 事件 × 4 检测器)")
    print("=" * 78)
    print(f"{'event':<12}" + "".join(f"{DET_EN[d]:<26}" for d in DETECTORS))
    print("-" * 78)
    for e in order:
        print(f"{e:<12}" + "".join(f"{matrix[e][d]:<26.3f}" for d in DETECTORS))
    print("-" * 78)
    for d in DETECTORS:
        vals = [matrix[e][d] for e in order]
        print(f"  {DET_EN[d]:<24}  mean AUROC={np.mean(vals):.3f}   min AUROC={np.min(vals):.3f}")

    print("\n绘图 ...")
    fig_bars(matrix)
    fig_table(matrix)
    print("-" * 78)
    print("已保存: results/fusion_ablation.json / _bars.png / _table.png")


if __name__ == "__main__":
    main()
