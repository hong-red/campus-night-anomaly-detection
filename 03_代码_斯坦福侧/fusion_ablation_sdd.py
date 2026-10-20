#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三路消融：硬性 if 规则 vs 深度学习(TCN) vs 相加融合 —— SDD 版「表 1」
=====================================================================

对应 F:\\仿真跑\\tcn_dl_branch\\fusion_ablation.py 的斯坦福版本。在真实 SDD 俯拍行人
轨迹上对 7 类事件做统一消融，证明「软性条件 + 硬性条件相加」能同时覆盖两类事件。

事件分类（表 1）
----------------
  规则域事件（EVT/规则分支负责，静止/空间/共现对 TCN 不可见）:
    dwell 长时间停留 / geofence 地理围栏突破 / gather 非工作时间聚集
  运动模式事件（深度学习分支负责，max(两个分支)）:
    wandering 徘徊 / sprint 冲刺 / zigzag / sudden-stop 突然停止

四个检测器
----------
  rules   硬性 if 规则 = max(dwell, gather, geofence)
  dl      深度学习 = z-clip(逐帧 ADE)，真零地板即置信度门限
  fused   相加 = max(rules, dl)              （软 + 硬，本实验核心）
  fluxev  完整 FluxEV 管线 = max(运动EVT, 规则)（方案1 基线）

额外（突停置信门控）: naive=max(fluxev, dl) vs gated=max(fluxev, dl·[dl>=0.5])，
证明「整段取 S=max(S_ev, S_dl) 会损害突停检测，需在 S_dl 上设置信度门限」。

输出 (results/)：
  sdd_fusion_ablation.json         7 事件 × 检测器 AUROC 矩阵 + 突停门控
  sdd_fusion_ablation_bars.png     分组条形图(AUROC，0.5 随机线)
  sdd_fusion_ablation_table.png    表 1(事件 → 负责分支 → AUROC)
  sdd_sudden_stop_gate.png         突停：S_ev / naive-max / gated 门控对照
"""
import os
import json

import sdd_common as S
import tcn_branch as T

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import metrics

RESULTS = S.RESULTS_DIR

DETECTORS = ["rules", "dl", "fused", "fluxev"]
DET_EN = {"rules": "Hard if-rules",
          "dl": "Deep learning (TCN)",
          "fused": "Fused (rules + DL)",
          "fluxev": "FluxEV pipeline (EVT + rules)"}
DET_COLOR = {"rules": "#1B7837", "dl": "#eb6834",
             "fused": "#0b0b0b", "fluxev": "#9AA0A6"}


# --------------------------- 事件注入 ---------------------------

def inject_event(x, y, event, dt=1.0):
    """在轨迹上注入 7 类事件之一，返回 (xa, ya, gt, others, zone)。"""
    n = len(x)
    xa, ya = x.copy(), y.copy()
    gt = np.zeros(n)
    others, zone = None, None
    i0 = int(n * 0.55)

    if event == "dwell":                    # 长时间停留：40s 原地不动(> DWELL_HI)
        i1 = min(i0 + 40, n - 2)
        xa[i0:i1] = xa[i0]; ya[i0:i1] = ya[i0]
        gt[i0:i1] = 1
    elif event == "sudden-stop":            # 突然停止后驻留 30s(TCN 见减速历史预测"原地"→盲区)
        i1 = min(i0 + 30, n - 2)
        xa[i0:i1] = xa[i0]; ya[i0:i1] = ya[i0]
        gt[i0:i1] = 1
    elif event == "wandering":              # 徘徊：来回折返
        i1 = min(i0 + max(16, int(n * 0.18)), n - 2)
        T.inject_loiter(xa, ya, i0, i1); gt[i0:i1] = 1
    elif event == "sprint":                 # 冲刺：位移放大
        i1 = min(i0 + max(16, int(n * 0.18)), n - 2)
        T.inject_run(xa, ya, i0, i1); gt[i0:i1] = 1
    elif event == "zigzag":                 # 蛇形：横向高频摆动
        i1 = min(i0 + max(16, int(n * 0.18)), n - 2)
        T.inject_zigzag(xa, ya, i0, i1); gt[i0:i1] = 1
    elif event == "gather":                 # 聚集：2 个"他人"贴靠 30s
        i1 = min(i0 + 30, n - 10)
        others = []
        for dx, dy in ((3.0, 0.0), (0.0, 3.0)):
            ox = np.full(n, xa[i0] + 20.0 + dx)
            oy = np.full(n, ya[i0] + 20.0 + dy)
            ox[i0:i1] = xa[i0:i1] + dx
            oy[i0:i1] = ya[i0:i1] + dy
            others.append((ox, oy))
        gt[i0:i1] = 1
    elif event == "geofence":               # 围栏：轨迹穿过 6m 圆(圆心在行进路径上)
        c = min(int(n * 0.7), n - 1)
        zone = (float(xa[c]), float(ya[c]), 6.0)
        gt = (S.geofence_score_2d(xa, ya, zone) > 0).astype(int)
    return xa, ya, gt, others, zone


def score_event(model, stats, traj, event, obs, pred):
    """注入事件并组装所有检测器分数，返回 (gt, {det: score})。"""
    sc, x, y, t = traj
    mu, sigma = stats["mu"], stats["sigma"]
    xa, ya, gt, others, zone = inject_event(x, y, event, 1.0)
    res = S.run_sdd_pipeline(xa, ya, t, 1.0, others=others, zone=zone)
    rules = np.maximum.reduce([res["dwell"], res["gather"], res["geofence"]])
    dl = S.zclip(S.frame_ade(model, xa, ya, obs, pred), mu, sigma)
    fluxev = res["final"]
    fused = np.maximum(rules, dl)
    naive = np.maximum(fluxev, dl)                 # 整段 S=max(S_ev, S_dl)
    gated = np.maximum(fluxev, dl * (dl >= 0.5))   # S_dl 置信度门限(>=0.5 才放行)
    return gt, {"rules": rules, "dl": dl, "fused": fused, "fluxev": fluxev,
                "naive": naive, "gated": gated}


def run_benchmark(model, stats, trajs, events, obs, pred):
    """逐事件、逐轨迹 AUROC 求均值，返回 {event: {det: auroc}}。"""
    agg = {(en, d): [] for en in events for d in DETECTORS + ["naive", "gated"]}
    for en in events:
        for sc, x, y, t in trajs:
            if len(x) < 60:
                continue
            gt, scs = score_event(model, stats, (sc, x, y, t), en, obs, pred)
            mask = np.arange(len(gt)) >= len(gt) // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            for d in DETECTORS + ["naive", "gated"]:
                agg[(en, d)].append(S._auroc(g, scs[d][mask]))
    return {en: {d: round(float(np.mean(agg[(en, d)])), 3)
                 for d in DETECTORS + ["naive", "gated"]} for en in events}


# --------------------------- 绘图 ---------------------------

def fig_bars(matrix):
    order = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag", "sudden-stop"]
    en_label = {"dwell": "Dwell\n(long stay)", "geofence": "Geofence\n(intrusion)",
                "gather": "Gather\n(>=3 ppl)", "wandering": "Wandering",
                "sprint": "Sprint", "zigzag": "Zigzag", "sudden-stop": "Sudden\nstop"}
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    x = np.arange(len(order)); w = 0.19
    for j, d in enumerate(DETECTORS):
        off = -1.5 * w + j * w
        vals = [matrix[e][d] for e in order]
        ax.bar(x + off, vals, w, color=DET_COLOR[d], zorder=3, label=DET_EN[d])
    ax.axhline(0.5, color="#c3c2b7", lw=1.0, ls="--", zorder=2)
    ax.text(len(order) - 0.5, 0.505, "chance 0.5", ha="right", va="bottom",
            fontsize=7.5, color="#898781")
    ax.set_xticks(x); ax.set_xticklabels([en_label[e] for e in order], fontsize=7.5)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Ablation across 7 real-trajectory events (Stanford Drone): "
                 "if-rules / DL / fused", loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="y", color="#e1e0d9", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sdd_fusion_ablation_bars.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def fig_table(matrix):
    def v(e, d):
        return f"{matrix[e][d]:.2f}"
    rows = [
        ["Event", "Responsible branch", "rules", "DL", "fused", "fluxev"],
        ["Dwell (long stay)", "EVT / if-rule (dwell)", v("dwell", "rules"),
         v("dwell", "dl"), v("dwell", "fused"), v("dwell", "fluxev")],
        ["Geofence intrusion", "EVT / if-rule (geofence)", v("geofence", "rules"),
         v("geofence", "dl"), v("geofence", "fused"), v("geofence", "fluxev")],
        ["Off-hours gathering", "EVT / if-rule (gather)", v("gather", "rules"),
         v("gather", "dl"), v("gather", "fused"), v("gather", "fluxev")],
        ["Wandering", "max(rules, DL)", v("wandering", "rules"),
         v("wandering", "dl"), v("wandering", "fused"), v("wandering", "fluxev")],
        ["Sprint", "max(rules, DL)", v("sprint", "rules"),
         v("sprint", "dl"), v("sprint", "fused"), v("sprint", "fluxev")],
        ["Zigzag", "max(rules, DL)", v("zigzag", "rules"),
         v("zigzag", "dl"), v("zigzag", "fused"), v("zigzag", "fluxev")],
        ["Sudden stop", "EVT motion + S_dl gate", v("sudden-stop", "rules"),
         v("sudden-stop", "dl"), v("sudden-stop", "fused"), v("sudden-stop", "fluxev")],
    ]
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.axis("off")
    tab = ax.table(cellText=rows, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.2)
    tab.scale(1, 1.6)
    for j in range(len(rows[0])):
        tab[(0, j)].set_text_props(fontweight="bold")
    fig.suptitle("Table 1 — event taxonomy, responsible branch, and AUROC (Stanford Drone)",
                 fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sdd_fusion_ablation_table.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def fig_sudden_stop_gate(matrix):
    """突停：S_ev(fluxev) vs naive-max(整段取 max) vs gated(置信门限)。"""
    ss = matrix["sudden-stop"]
    labels = ["FluxEV\n(S_ev)", "naive max\n(S_ev, S_dl)", "gated\n(S_dl>=0.5)"]
    vals = [ss["fluxev"], ss["naive"], ss["gated"]]
    colors = ["#9AA0A6", "#e34948", "#1B7837"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(3)
    ax.bar(x, vals, 0.55, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="#c3c2b7", ls="--", lw=0.9, zorder=2)
    ax.text(2.4, 0.505, "chance 0.5", ha="right", va="bottom", fontsize=7.5, color="#898781")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC (sudden-stop)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Sudden stop: naive max hurts detection, confidence gate restores it",
                 loc="left", fontsize=9)
    ax.grid(axis="y", color="#e1e0d9", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sdd_sudden_stop_gate.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def main():
    model, stats, cfg = S.ensure_sdd_tcn()
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    print(f"模型: {S.SDD_TCN_CKPT}  (obs={obs}, pred={pred}, mu={mu:.2f}m, sigma={sigma:.2f}m)")

    trajs = S.collect_sdd_walking(1.0, 150, seed=7, min_len=60)
    print(f"评估行走轨迹数 = {len(trajs)}")

    events = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag", "sudden-stop"]
    matrix = run_benchmark(model, stats, trajs, events, obs, pred)

    json.dump(matrix, open(os.path.join(RESULTS, "sdd_fusion_ablation.json"), "w"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 82)
    print("AUROC 矩阵  (7 事件 × 检测器)")
    print("=" * 82)
    print(f"{'event':<12}" + "".join(f"{DET_EN[d]:<30}" for d in DETECTORS) + "naive   gated")
    print("-" * 82)
    for e in events:
        print(f"{e:<12}" + "".join(f"{matrix[e][d]:<30.3f}" for d in DETECTORS)
              + f"{matrix[e]['naive']:<8.3f}{matrix[e]['gated']:.3f}")
    print("-" * 82)
    for d in DETECTORS:
        vals = [matrix[e][d] for e in events]
        print(f"  {DET_EN[d]:<28}  mean={np.mean(vals):.3f}  min={np.min(vals):.3f}")

    print("\n绘图 ...")
    fig_bars(matrix)
    fig_table(matrix)
    fig_sudden_stop_gate(matrix)
    print("-" * 82)
    print("已保存: results/sdd_fusion_ablation.json / _bars.png / _table.png / "
          "sdd_sudden_stop_gate.png")


if __name__ == "__main__":
    main()
