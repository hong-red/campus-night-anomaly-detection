#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DL 分支(TCN 轨迹预测) 在 Stanford Drone Dataset 上对「原地逗留」与「徘徊」的
准确率 / 召回率实验
===============================================================================

对应 F:\\仿真跑\\tcn_dl_branch\\eval_dwell_wander.py 的斯坦福版本：TCN 架构与训练目标
完全相同(自监督轨迹预测)，仅在 SDD 正常行走轨迹上重训(dt=1s)。

  徘徊 (wandering)      —— 沿行进方向来回折返。偏离"正常人直行"先验 → 预测误差大 → 可检。
  原地逗留 (loitering)  —— 停在原地不动。TCN 看到减速历史后预测"原地"=真实 → 预测误差小 → 盲区。

命名对齐：tcn_branch.inject_loiter = 徘徊(来回折返)，inject_stop = 原地逗留(驻留不动)。

指标（窗口级，ADE 作为连续分数）：
  - AUROC / AP(PR 面积)        阈值无关判别力
  - @最优 F1 阈值              精确率 / 召回率 / F1 / 准确率 / 平衡准确率 / 混淆矩阵
  - @S_dl>=0.5                 项目固定报警阈值(sigmoid 标定)

输出 (results/)：
  sdd_dl_dwell_wander.json    全量指标
  sdd_dl_roc_pr.png           ROC + PR 曲线(两行为叠加)
  sdd_dl_metrics_bars.png     精确率/召回率/F1/准确率 分组条形图
  sdd_dl_demo.png             同一条轨迹上 徘徊(检出) vs 原地逗留(漏检) 逐帧 S_dl 对照
"""
import os
import sys
import json

import sdd_common as S
import tcn_branch as T

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import metrics

RESULTS = S.RESULTS_DIR

C_WANDER = S.C_WANDER
C_LOITER = S.C_LOITER
C_CHANCE = S.C_CHANCE
C_GT = S.C_GT
C_MUTED = S.C_MUTED
C_GRID = S.C_GRID

BEHAVIORS = [
    ("wandering", "Wandering",          T.inject_loiter, C_WANDER),
    ("loitering", "Loitering (in place)", T.inject_stop, C_LOITER),
]
ZH = {"wandering": "徘徊", "loitering": "原地逗留"}


def labeled_ade(model, trajs, injector, obs, pred, stride):
    total = obs + pred
    ades, labels = [], []
    for (sc, x, y, t) in trajs:
        n = len(x)
        if n < total + 24:
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + max(16, int(n * 0.18)), n - total - 2)
        if i1 - i0 < 8:
            continue
        xa, ya = x.copy(), y.copy()
        mask = injector(xa, ya, i0, i1)
        obs_list, fut_list, lab_list = [], [], []
        for s in range(0, n - total + 1, stride):
            fut_seg = np.arange(s + obs, s + total)
            lab_list.append(1 if np.isin(fut_seg, mask).any() else 0)
            ref = np.array([xa[s + obs - 1], ya[s + obs - 1]])
            obs_list.append(np.stack([xa[s:s + obs], ya[s:s + obs]], -1) - ref)
            fut_list.append(np.stack([xa[s + obs:s + total], ya[s + obs:s + total]], -1) - ref)
        ades.append(T.compute_ade(model, np.asarray(obs_list, np.float32),
                                  np.asarray(fut_list, np.float32)))
        labels.append(np.asarray(lab_list, dtype=int))
    return np.concatenate(ades), np.concatenate(labels)


def binary_metrics(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(gt) if len(gt) else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    bacc = (tpr + tnr) / 2
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "balanced_accuracy": bacc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_behavior(model, trajs, obs, pred, stride, name, stats):
    ade, gt = labeled_ade(model, trajs, T.inject_loiter if name == "wandering"
                          else T.inject_stop, obs, pred, stride)
    auroc = float(metrics.roc_auc_score(gt, ade))
    ap = float(metrics.average_precision_score(gt, ade))

    prec_c, rec_c, thr_c = metrics.precision_recall_curve(gt, ade)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)
    k = int(np.argmax(f1_c))
    best_thr = float(thr_c[k]) if k < len(thr_c) else float(ade.max())
    best = binary_metrics((ade >= best_thr).astype(int), gt)
    best["threshold_ade"] = round(best_thr, 3)

    s_dl = T.ade_to_score(ade, stats["mu"], stats["sigma"], stats.get("offset", 2.0))
    at05 = binary_metrics((s_dl >= 0.5).astype(int), gt)

    return {
        "auroc": round(auroc, 4),
        "ap": round(ap, 4),
        "functional": bool(auroc >= 0.5),
        "n_windows": int(len(gt)),
        "n_pos": int(gt.sum()), "n_neg": int((1 - gt).sum()),
        "best_f1_threshold": best,
        "at_Sdl_0_5": {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                       for kk, vv in at05.items()},
        "pr_curve": {"precision": [round(v, 4) for v in prec_c],
                     "recall": [round(v, 4) for v in rec_c]},
        "_ade": ade, "_gt": gt,
    }


def demo_frame_scores(model, traj, injector, obs, pred, stats):
    sc, x, y, t = traj
    n = len(x)
    i0, i1 = int(n * 0.55), min(int(n * 0.75), n - obs - pred - 2)
    xa, ya = x.copy(), y.copy()
    injector(xa, ya, i0, i1)
    s_dl = T.tcn_frame_scores(model, xa, ya, obs, pred, stats, stride=1)
    s_dl = np.nan_to_num(s_dl, nan=0.0)
    return s_dl, i0, i1


def fig_roc_pr(res):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    ax = axes[0]
    for key, label, _inj, color in BEHAVIORS:
        ade, gt = res[key]["_ade"], res[key]["_gt"]
        fpr, tpr, _ = metrics.roc_curve(gt, ade)
        ax.plot(fpr, tpr, color=color, lw=1.6,
                label=f"{label} (AUROC={res[key]['auroc']:.2f})")
    ax.plot([0, 1], [0, 1], color=C_CHANCE, lw=1.0, ls="--")
    ax.text(0.98, 0.04, "chance", color=C_MUTED, fontsize=7, ha="right")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — TCN (DL) per window, Stanford Drone", loc="left", fontsize=9)
    ax.grid(True, color=C_GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for key, label, _inj, color in BEHAVIORS:
        p = np.array(res[key]["pr_curve"]["precision"])
        r = np.array(res[key]["pr_curve"]["recall"])
        ax.plot(r, p, color=color, lw=1.6, label=f"{label} (AP={res[key]['ap']:.2f})")
        bf = res[key]["best_f1_threshold"]
        ax.scatter([bf["recall"]], [bf["precision"]], color=color, s=34, zorder=5,
                   edgecolor="white", linewidth=0.7)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR — dots = best-F1 operating point", loc="left", fontsize=9)
    ax.grid(True, color=C_GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sdd_dl_roc_pr.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_bars(res):
    metrics_keys = [("precision", "Precision"), ("recall", "Recall"),
                    ("f1", "F1"), ("accuracy", "Accuracy"),
                    ("balanced_accuracy", "Balanced acc.")]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = np.arange(len(metrics_keys)); w = 0.30; gap = 0.06
    for j, (key, label, _inj, color) in enumerate(BEHAVIORS):
        vals = [res[key]["best_f1_threshold"][m] for m, _ in metrics_keys]
        off = -w / 2 - gap / 2 + j * (w + gap)
        ax.bar(x + off, vals, w, color=color, zorder=3,
               label=f"{label} (AUROC {res[key]['auroc']:.2f})")
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color=C_CHANCE, lw=0.9, ls="--", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([m for _, m in metrics_keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("TCN (DL) @ best-F1 threshold: wandering vs loitering-in-place (SDD)",
                 loc="left", fontsize=9)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=C_GRID, lw=0.5); ax.set_axisbelow(True)
    if not res["loitering"]["functional"]:
        fig.text(0.5, 0.01,
                 "Loitering-in-place: AUROC < 0.5 → its 'best-F1' point degenerates to "
                 "predict-all-positive (recall=1, precision=prior); no usable operating point.",
                 ha="center", va="bottom", fontsize=7.0, color=C_MUTED)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sdd_dl_metrics_bars.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_demo(model, traj, obs, pred, stats):
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True)
    sc, x, y, t = traj
    tt = t
    for ax, (key, label, inj, color) in zip(axes, BEHAVIORS):
        s_dl, i0, i1 = demo_frame_scores(model, traj, inj, obs, pred, stats)
        ax.fill_between(tt, 0, 1, where=(np.arange(len(tt)) >= i0) & (np.arange(len(tt)) < i1),
                        color=C_GT, alpha=0.12, lw=0, label="injected anomaly")
        ax.plot(tt, s_dl, color=color, lw=1.3, label=f"TCN S$_{{dl}}$ ({label})")
        ax.axhline(0.5, color=C_CHANCE, lw=0.9, ls="--")
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("S$_{dl}$")
        ax.set_title(f"{label} — " + ("detected" if key == "wandering" else "missed (structural blind spot)"),
                     loc="left", fontsize=9)
        ax.legend(frameon=False, loc="upper right")
        ax.grid(True, color=C_GRID, lw=0.5)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sdd_dl_demo.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    model, stats, cfg = S.ensure_sdd_tcn()
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    print(f"模型: {S.SDD_TCN_CKPT}  (obs={obs}, pred={pred}, mu={stats['mu']:.2f}m, "
          f"sigma={stats['sigma']:.2f}m)")

    trajs = S.collect_sdd_walking(1.0, 250, seed=42, min_len=48)
    print(f"评估行走轨迹数 = {len(trajs)}")

    demo_traj = max(trajs, key=lambda t: len(t[1]))

    print("\n" + "=" * 74)
    print("窗口级评估：AUROC / AP / @最优F1阈值 / @S_dl>=0.5")
    print("=" * 74)
    res = {}
    for key, label, inj, _color in BEHAVIORS:
        r = evaluate_behavior(model, trajs, obs, pred, 2, key, stats)
        res[key] = r
        bf = r["best_f1_threshold"]; a05 = r["at_Sdl_0_5"]
        flag = "" if r["functional"] else "  [非判别方向: AUROC<0.5, 结构盲区]"
        print(f"\n[{ZH[key]} / {label}]  n_window={r['n_windows']} "
              f"(pos={r['n_pos']}, neg={r['n_neg']}){flag}")
        print(f"  AUROC={r['auroc']:.3f}   AP={r['ap']:.3f}")
        print(f"  @最优F1阈值(ADE>={bf['threshold_ade']:.2f}m): "
              f"P={bf['precision']:.3f}  R={bf['recall']:.3f}  F1={bf['f1']:.3f}  "
              f"acc={bf['accuracy']:.3f}  bacc={bf['balanced_accuracy']:.3f}  "
              f"[TP={bf['tp']} FP={bf['fp']} FN={bf['fn']} TN={bf['tn']}]")
        print(f"  @S_dl>=0.5(默认标定):  P={a05['precision']:.3f}  R={a05['recall']:.3f}  "
              f"F1={a05['f1']:.3f}  acc={a05['accuracy']:.3f}   ← 噪声地板导致召回塌陷")

    json_out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in res.items()}
    json.dump(json_out, open(os.path.join(RESULTS, "sdd_dl_dwell_wander.json"), "w"),
              ensure_ascii=False, indent=2)

    print("\n绘图 ...")
    fig_roc_pr(res)
    fig_metrics_bars(res)
    fig_demo(model, demo_traj, obs, pred, stats)
    print("-" * 74)
    print("已保存: results/sdd_dl_dwell_wander.json / sdd_dl_roc_pr.png / "
          "sdd_dl_metrics_bars.png / sdd_dl_demo.png")


if __name__ == "__main__":
    main()
