#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强约束 if 规则(禁区闯入 / 多人聚集) 在 Stanford Drone Dataset 上的准确率 + 优势实验
=================================================================================

对应 F:\\仿真跑\\experiment_rule_advantage.py 的斯坦福版本：把"合成轨迹"换成 SDD 真实
俯拍行人轨迹(像素 -> 米)，其余论证结构一致：

  1) 禁区闯入 —— if(inside_circle)：真实行人匀速穿越场景内一块虚拟圆形禁区，
     真值=几何内外(多边形/圆定义)，规则判定 vs FluxEV 运动分数做对照。
     论据：穿越围栏不改变运动模式，运动模型对"内外"无判别力。
  2) 多人聚集 —— if(>=3人 / 5m)：真实主行人匀速直行，2 个"他人"在聚集时段贴靠(3m)，
     真值=聚集时段，规则判定 vs FluxEV 运动分数做对照。
     论据：主行人全程匀速(运动分数恒 0)，聚集是跨轨迹空间共现，运动模型看不到。

输出 (results/)：
  sdd_rule_advantage_geofence.png   禁区闯入(轨迹 + 时序：规则报警 vs 运动分数)
  sdd_rule_advantage_gather.png     多人聚集(距离 + 时序：规则报警 vs 运动分数)
  sdd_rule_advantage_summary.png    规则 vs 运动模型 汇总条形图
  sdd_rule_advantage.json           全量指标
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import sdd_common as S
from sdd_common import (C_RULE, C_FLUXEV, C_GT, C_MUTED, C_CHANCE,
                        GATHER_RADIUS, GATHER_MIN_PEOPLE, ZONE_RADIUS,
                        run_sdd_pipeline, geofence_score_2d, gather_score_2d)

os.makedirs(S.RESULTS_DIR, exist_ok=True)


def auroc(gt, score):
    from sklearn import metrics
    if (gt == 1).any() and (gt == 0).any() and np.std(score) > 1e-9:
        return float(metrics.roc_auc_score(gt, score))
    return 0.5


def find_crossing_tracks(scene, radius=ZONE_RADIUS, min_frac=0.15, max_frac=0.85):
    """在该场景找"穿越圆形禁区"的真实行走轨迹，返回 [(x, y, t, cx, cy, inside)]。"""
    tracks = S.collect_sdd_walking(1.0, None, seed=0, min_len=48)
    tracks = [t for t in tracks if t[0] == scene]
    if not tracks:
        return []
    xs = np.concatenate([t[1] for t in tracks])
    ys = np.concatenate([t[2] for t in tracks])
    cx, cy = float(np.median(xs)), float(np.median(ys))
    out = []
    for _, x, y, t in tracks:
        inside = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        frac = float(inside.mean())
        if not (min_frac <= frac <= max_frac):
            continue
        if inside[0] or inside[-1]:        # 起终点在圈外才叫"闯入后离开"
            continue
        out.append((x, y, t, cx, cy, inside))
    return out


def run_geofence():
    print("=" * 78)
    print("[1/2] 禁区闯入 —— if(inside_circle)  vs  FluxEV motion")
    print("=" * 78)
    # 跨场景累计：找所有"穿越"轨迹，逐轨迹算运动模型 AUROC(规则几何精确 -> 恒 1.0)
    scenes = sorted(set(t[0] for t in S.load_sdd_ped_tracks()))
    cross_all, demo = [], None
    for sc in scenes:
        for x, y, t, cx, cy, inside in find_crossing_tracks(sc):
            cross_all.append((sc, x, y, t, cx, cy, inside))
    if not cross_all:
        print("  未找到穿越轨迹，改用单一场景中心圆兜底 ...")
        return
    rule_acc, motion_aucs = [], []
    for sc, x, y, t, cx, cy, inside in cross_all:
        gt = inside.astype(float)
        rule = geofence_score_2d(x, y, (cx, cy, ZONE_RADIUS))
        res = run_sdd_pipeline(x, y, t, 1.0)
        rule_acc.append(float((rule == gt).mean()))
        motion_aucs.append(auroc(gt, res["motion"]))
        if demo is None:
            demo = (sc, x, y, t, cx, cy, inside, res["motion"])
    geo_acc = float(np.mean(rule_acc))
    geo_motion_auc = float(np.mean(motion_aucs))
    n_cross = len(cross_all)
    print(f"  穿越轨迹数 = {n_cross}（跨 {len(set(c[0] for c in cross_all))} 个场景）")
    print(f"  规则 accuracy = {geo_acc:.4f}（几何精确） |  运动模型 AUROC = {geo_motion_auc:.3f}")

    # 演示图：轨迹(圈内外着色) + 时序(规则报警 vs 运动分数)
    sc, x, y, t, cx, cy, inside, motion = demo
    th = np.linspace(0, 2 * np.pi, 120)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(cx + ZONE_RADIUS * np.cos(th), cy + ZONE_RADIUS * np.sin(th),
            ls="--", lw=1.6, color=C_GT, label="Restricted zone")
    ax.fill(cx + ZONE_RADIUS * np.cos(th), cy + ZONE_RADIUS * np.sin(th),
            color=C_GT, alpha=0.08)
    ax.plot(x[~inside], y[~inside], lw=1.0, color="0.6", label="Outside")
    ax.plot(x[inside], y[inside], lw=2.0, color=C_GT, label="Inside (intrusion)")
    ax.scatter(x[0], y[0], s=50, color="#00B140", zorder=5, label="Start")
    ax.scatter(x[-1], y[-1], s=50, color="#0B6BFF", zorder=5, label="End")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{sc} — a real pedestrian crosses the fence (constant-speed walk)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="best"); ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    rule = inside.astype(float)
    ax.step(t, rule, where="post", color=C_RULE, lw=1.4, label="if(inside-circle)")
    ax.plot(t, motion, color=C_FLUXEV, lw=1.3, label="FluxEV motion score")
    ax.axhline(0.5, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Score / alarm")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"Rule exact (acc={geo_acc:.2f}) vs motion blind (AUROC={geo_motion_auc:.2f})",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(S.RESULTS_DIR, "sdd_rule_advantage_geofence.png"), dpi=200)
    plt.close(fig)
    return dict(n_cross=n_cross, rule_acc=geo_acc, motion_auc=geo_motion_auc)


def run_gather():
    print("=" * 78)
    print("[2/2] 多人聚集 —— if(>=3人 / 5m)  vs  FluxEV motion")
    print("=" * 78)
    tracks = S.collect_sdd_walking(1.0, 120, seed=7, min_len=60)
    dt = 1.0
    gt_all, alarm_all, score_all = [], [], []
    motion_aucs, rule_aucs = [], []
    demo = None
    for sc, x, y, t in tracks:
        n = len(x)
        if n < 70:
            continue
        g0 = int(n * 0.55)
        g1 = min(g0 + 30, n - 10)          # 30s 聚集时段
        if g1 - g0 < 15:
            continue
        # 2 个"他人"：平时远离 20m，聚集时段贴靠主行人(3m)
        others = []
        for dx, dy in ((3.0, 0.0), (0.0, 3.0)):
            ox = np.full(n, x[g0] + 20.0 + dx)
            oy = np.full(n, y[g0] + 20.0 + dy)
            ox[g0:g1] = x[g0:g1] + dx
            oy[g0:g1] = y[g0:g1] + dy
            others.append((ox, oy))
        gt = np.zeros(n); gt[g0:g1] = 1
        score = gather_score_2d(x, y, others, dt)
        alarm = (score >= 0.5).astype(int)
        res = run_sdd_pipeline(x, y, t, dt)   # 主行人轨迹不变 -> 运动模型盲
        gt_all.append(gt); alarm_all.append(alarm); score_all.append(score)
        motion_aucs.append(auroc(gt, res["motion"]))
        rule_aucs.append(auroc(gt, score))
        if demo is None:
            demo = (sc, x, y, t, others, gt, score, alarm, res["motion"], g0, g1)

    gt = np.concatenate(gt_all); alarm = np.concatenate(alarm_all)
    score = np.concatenate(score_all)
    tp = int(((alarm == 1) & (gt == 1)).sum())
    fp = int(((alarm == 1) & (gt == 0)).sum())
    fn = int(((alarm == 0) & (gt == 1)).sum())
    tn = int(((alarm == 0) & (gt == 0)).sum())
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec); acc = (tp + tn) / len(gt)
    g_motion_auc = float(np.mean(motion_aucs))
    g_rule_auc = float(np.mean(rule_aucs))
    print(f"  主行人轨迹数 = {len(gt_all)}，聚集帧占比 = {gt.mean()*100:.1f}%")
    print(f"  规则 P={prec:.3f} R={rec:.3f} F1={f1:.3f} acc={acc:.3f} AUROC={g_rule_auc:.3f}"
          f"  |  运动模型 AUROC = {g_motion_auc:.3f}")

    # 演示图：距离 + 时序
    sc, x, y, t, others, gt, s, alarm, motion, g0, g1 = demo
    (ox0, oy0), (ox1, oy1) = others
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    ax = axes[0]
    ax.plot(t, np.hypot(ox0 - x, oy0 - y), color=C_GT, lw=0.9, label="gatherer A distance")
    ax.plot(t, np.hypot(ox1 - x, oy1 - y), color=C_GT, lw=0.9, ls="-.", label="gatherer B distance")
    ax.axhline(GATHER_RADIUS, color="#FF7A00", ls="--", lw=0.9, label=f"{GATHER_RADIUS} m radius")
    ax.axvspan(g0 * dt, g1 * dt, color="#FF7A00", alpha=0.12, label="Gathering period (GT)")
    ax.set_ylabel("distance to main (m)")
    ax.set_title(f"{sc} — gathering: proximity over time", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.step(t, alarm, where="post", color=C_RULE, lw=1.4, label="if(>=3 people / 5 m)")
    ax.plot(t, motion, color=C_FLUXEV, lw=1.2, label="FluxEV motion score (main person)")
    ax.axvspan(g0 * dt, g1 * dt, color="#FF7A00", alpha=0.12, label="Gathering period (GT)")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("score / alarm")
    ax.set_title(f"Rule P={prec:.2f} R={rec:.2f} F1={f1:.2f}  vs  motion blind (AUROC={g_motion_auc:.2f})",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(S.RESULTS_DIR, "sdd_rule_advantage_gather.png"), dpi=200)
    plt.close(fig)
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc,
                rule_auc=g_rule_auc, motion_auc=g_motion_auc, n=len(gt_all))


def main():
    geo = run_geofence()
    gat = run_gather()

    out = {
        "geofence_intrusion": {
            "rule": {"accuracy": round(geo["rule_acc"], 4), "auroc": 1.0},
            "motion_fluxev": {"auroc": round(geo["motion_auc"], 4)},
            "n_crossing_tracks": geo["n_cross"],
        },
        "multi_person_gathering": {
            "rule": {k: round(v, 4) for k, v in gat.items()
                     if k in ("precision", "recall", "f1", "accuracy", "rule_auc")},
            "motion_fluxev": {"auroc": round(gat["motion_auc"], 4)},
            "n_tracks": gat["n"],
        },
    }
    json.dump(out, open(os.path.join(S.RESULTS_DIR, "sdd_rule_advantage.json"), "w"),
              ensure_ascii=False, indent=2)

    # 汇总条形图：规则 vs 运动模型
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    tasks = ["Geofence\nintrusion", "Multi-person\ngathering"]
    rule_auc = [1.0, gat["rule_auc"]]
    motion_auc = [geo["motion_auc"], gat["motion_auc"]]
    x = np.arange(2); w = 0.34
    ax.bar(x - w / 2, rule_auc, w, color=C_RULE, label="if-rule", zorder=3)
    ax.bar(x + w / 2, motion_auc, w, color=C_FLUXEV, label="FluxEV motion (EVT)", zorder=3)
    for xi, v in zip(x - w / 2, rule_auc):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, motion_auc):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=C_CHANCE, ls="--", lw=0.9)
    ax.text(1.17, 0.515, "chance 0.5", ha="right", va="bottom", fontsize=7.5, color="0.4")
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.15)
    ax.set_title("Hard if-conditions detect spatial properties motion models cannot",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(S.RESULTS_DIR, "sdd_rule_advantage_summary.png"), dpi=200)
    plt.close(fig)

    print("-" * 78)
    print("已保存: results/sdd_rule_advantage_geofence.png / _gather.png / _summary.png / "
          "sdd_rule_advantage.json")


if __name__ == "__main__":
    main()
