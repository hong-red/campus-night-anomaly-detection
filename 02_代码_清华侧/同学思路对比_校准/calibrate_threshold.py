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
"""报警阈值定标（calibration）：三种方法 + F1 说明图。

在验证集（TCN DL 分支，窗口级分数 S_dl ∈ [0,1]）上，把「报警阈值 τ 怎么定」做成一张图：

  方法① F1 最优点（最常用）：PR 曲线上使 F1 = 2·P·R/(P+R) 最大的点 → 阈值 τ_F1
  方法② Youden J 指数：ROC 上使 J = TPR + TNR − 1 = TPR − FPR 最大的点 → τ_J
  方法③ 固定误报率：给定可接受 FPR（默认 5%）下取最大 TPR 的阈值 → τ_FPR

三张面板：
  (a) PR 曲线 + 等 F1 线，标 F1 最优点 vs 现行 0.5
  (b) F1 / P / R 随阈值 τ 的曲线，三条竖线标 τ_F1 / τ_J / τ_FPR，红色虚线 = 现行 0.5（报警点）
  (c) ROC 曲线 + Youden 最优点 + 5% FPR 点

以「徘徊 wandering」为演示（AUROC 0.77，功能正常）；
「原地逗留 loitering」AUROC 0.19 是结构盲区，无可用阈值，仅落盘备注、不画。

输出：
  threshold_calib.json        逐窗口 S_dl + 标签 + 三种阈值 + PR/ROC 曲线点
  figs/fig_threshold_calib.png

用法：python calibrate_threshold.py [--max_traj 250] [--seed 42] [--fpr 0.05]
"""

import os
import sys
import json
import argparse

# Windows 下必须先 import torch 再 import numpy（c10.dll 冲突，见 tcn_branch.py）
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = _PKGP.CODE_TSINGHUA
_TCN = os.path.join(_PROJ, "tcn_dl_branch")
for _p in (_TCN, _PROJ, _PKGP.STYLE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tcn_branch as T
from sklearn import metrics
from style import apply_nature_style, C, DOUBLE_COL, MM, panel_label, save

apply_nature_style()

INK = "#252525"
BLUE = C["calm"]     # 主曲线
GREEN = C["green"]   # 方法① F1 最优点
ORANGE = C["accent"] # 方法② Youden
GREY = C["grey"]     # 方法③ 固定 FPR / 参考线
RED = C["alert"]     # 现行 0.5 报警阈值


# --------------------------- 窗口级 (S_dl, 标签) ---------------------------

def labeled_ade(model, trajs, injector, obs, pred, stride):
    """每条轨迹注入异常段，滑窗切 (obs, fut)，窗口真值 = 未来段与注入区是否重叠，
    返回 (ADE, 标签)。与 eval_dwell_wander.py 同实现。"""
    total = obs + pred
    ades, labels = [], []
    for (x, y, *_rest) in trajs:
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
            lab_list.append(1 if mask[np.isin(mask, fut_seg)].size else 0)
            ref = np.array([xa[s + obs - 1], ya[s + obs - 1]])
            obs_list.append(np.stack([xa[s:s + obs], ya[s:s + obs]], -1) - ref)
            fut_list.append(np.stack([xa[s + obs:s + total], ya[s + obs:s + total]], -1) - ref)
        ades.append(T.compute_ade(model, np.asarray(obs_list, np.float32),
                                  np.asarray(fut_list, np.float32)))
        labels.append(np.asarray(lab_list, dtype=int))
    return np.concatenate(ades), np.concatenate(labels)


def metrics_at(gt, score, t):
    pred = (score >= t).astype(int)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(gt)
    return dict(precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
                accuracy=round(acc, 4), tp=tp, fp=fp, fn=fn, tn=tn)


def sweep(gt, score, n=600):
    """在分数区间上均匀扫阈值，返回 (τ, P, R, F1) 四组数组（画图用）。"""
    lo, hi = float(score.min()), float(score.max())
    ts = np.linspace(lo, hi, n)
    pos_score, neg_score = score[gt == 1], score[gt == 0]
    tp = np.array([(pos_score >= t).sum() for t in ts], dtype=float)
    fp = np.array([(neg_score >= t).sum() for t in ts], dtype=float)
    fn = gt.sum() - tp
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return ts, prec, rec, f1


def calibrate(gt, score, fpr_target=0.05):
    """三种定标方法，返回各自阈值 + 该阈值下的 P/R/F1。"""
    auroc = float(metrics.roc_auc_score(gt, score))
    ap = float(metrics.average_precision_score(gt, score))

    # ① F1 最优点（PR 曲线）
    prec, rec, thr = metrics.precision_recall_curve(gt, score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.argmax(f1))
    tau_f1 = float(thr[k]) if k < len(thr) else None          # k==len(thr) 即退化到"全报警"
    m_f1 = metrics_at(gt, score, tau_f1) if tau_f1 is not None else None

    # ② Youden J（ROC 曲线，J = TPR - FPR）
    fpr, tpr, thr_roc = metrics.roc_curve(gt, score)
    j = tpr - fpr
    kj = int(np.argmax(j))
    tau_j = float(thr_roc[kj])
    m_j = metrics_at(gt, score, tau_j)

    # ③ 固定误报率：FPR ≤ fpr_target 下取最大 TPR（最宽松且不超预算的阈值）
    ok = fpr <= fpr_target
    idx = int(np.where(ok)[0][-1]) if ok.any() else 0
    tau_fpr = float(thr_roc[idx])
    m_fpr = metrics_at(gt, score, tau_fpr)

    return {
        "auroc": round(auroc, 4), "ap": round(ap, 4),
        "f1_optimal":  {"threshold": round(tau_f1, 4) if tau_f1 is not None else None,
                        "metrics": m_f1},
        "youden_j":    {"threshold": round(tau_j, 4), "metrics": m_j},
        "fixed_fpr":   {"threshold": round(tau_fpr, 4), "fpr_target": fpr_target,
                        "metrics": m_fpr},
        "pr_curve":    {"precision": [round(v, 4) for v in prec],
                        "recall": [round(v, 4) for v in rec]},
        "roc_curve":   {"fpr": [round(v, 4) for v in fpr],
                        "tpr": [round(v, 4) for v in tpr],
                        "youden": [round(v, 4) for v in j]},
    }


# --------------------------- 绘图 ---------------------------

def fig_calibration(gt, score, cal, fpr_target):
    ts, prec_s, rec_s, f1_s = sweep(gt, score)
    tau_f1 = cal["f1_optimal"]["threshold"]
    tau_j = cal["youden_j"]["threshold"]
    tau_fpr = cal["fixed_fpr"]["threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 58 * MM))

    # ---- (a) PR 曲线 + 等 F1 线 ----
    ax = axes[0]
    prec, rec = np.array(cal["pr_curve"]["precision"]), np.array(cal["pr_curve"]["recall"])
    ax.plot(rec, prec, color=BLUE, lw=1.4, label="wandering")
    # 等 F1 线（P = F·R / (2R − F)）
    for fiso in (0.1, 0.3, 0.5, 0.7):
        rr = np.linspace(fiso / (2 - fiso) + 1e-3, 1.0, 200)
        ax.plot(rr, fiso * rr / (2 * rr - fiso), color=GREY, lw=0.5, ls=(0, (3, 3)), alpha=0.55)
        ax.text(0.985, fiso * 1 / (2 * 1 - fiso) + 0.015, f"F1={fiso:.1f}",
                ha="right", va="bottom", fontsize=6, color=GREY)
    if tau_f1 is not None:
        m1 = cal["f1_optimal"]["metrics"]
        ax.scatter([m1["recall"]], [m1["precision"]], s=42, color=GREEN, zorder=5,
                   marker="*", edgecolor="white", linewidth=0.7)
        ax.annotate(f"F1-max {m1['f1']:.2f}\n$\\tau$={tau_f1:.2f}",
                    (m1["recall"], m1["precision"]), xytext=(0.12, 0.30),
                    textcoords="axes fraction", fontsize=6.5, color=GREEN,
                    arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6))
    m05 = metrics_at(gt, score, 0.5)
    ax.scatter([m05["recall"]], [m05["precision"]], s=34, color=RED, zorder=5,
               edgecolor="white", linewidth=0.7)
    ax.annotate(f"$\\tau$=0.5 (current)\nF1={m05['f1']:.2f}",
                (m05["recall"], m05["precision"]), xytext=(0.52, 0.72),
                textcoords="axes fraction", fontsize=6.5, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("PR — method 1: max-F1", loc="left", pad=3)

    # ---- (b) F1 / P / R vs 阈值 τ（报警点） ----
    ax = axes[1]
    ax.plot(ts, f1_s, color=BLUE, lw=1.6, label="F1")
    ax.plot(ts, prec_s, color=GREY, lw=0.8, ls="--", label="Precision")
    ax.plot(ts, rec_s, color=GREY, lw=0.8, ls=":", label="Recall")
    ax.axvline(0.5, color=RED, ls="--", lw=1.1)
    ax.text(0.5, 0.985, "0.5\n(current)", ha="left", va="top", fontsize=6, color=RED)
    if tau_f1 is not None:
        ax.axvline(tau_f1, color=GREEN, lw=1.2)
        ax.text(tau_f1, 0.985, f"F1\n{tau_f1:.2f}", ha="left", va="top", fontsize=6, color=GREEN)
    ax.axvline(tau_j, color=ORANGE, lw=1.2)
    ax.text(tau_j, 0.55, f"J {tau_j:.2f}", ha="left", va="bottom", fontsize=6, color=ORANGE)
    ax.axvline(tau_fpr, color=GREY, lw=1.2)
    ax.text(tau_fpr, 0.28, f"FPR≤{fpr_target:.0%} {tau_fpr:.2f}",
            ha="left", va="bottom", fontsize=6, color=GREY)
    if tau_f1 is not None:
        kf = int(np.argmax(f1_s))
        ax.scatter([ts[kf]], [f1_s[kf]], s=30, color=GREEN, zorder=6, marker="*",
                   edgecolor="white", linewidth=0.6)
    ax.set_xlabel("alarm threshold $\\tau$ (score)")
    ax.set_ylabel("score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("F1 / P / R vs $\\tau$ — where to alarm", loc="left", pad=3)
    ax.legend(frameon=False, fontsize=6, loc="center right")

    # ---- (c) ROC 曲线 + Youden ----
    ax = axes[2]
    fpr, tpr, youden = (np.array(cal["roc_curve"]["fpr"]),
                        np.array(cal["roc_curve"]["tpr"]),
                        np.array(cal["roc_curve"]["youden"]))
    ax.plot(fpr, tpr, color=BLUE, lw=1.4, label=f"AUROC {cal['auroc']:.2f}")
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
    kj = int(np.argmax(youden))
    ax.scatter([fpr[kj]], [tpr[kj]], s=42, color=ORANGE, zorder=5, marker="*",
               edgecolor="white", linewidth=0.7)
    ax.annotate(f"Youden J={youden[kj]:.2f}\n$\\tau$={tau_j:.2f}",
                (fpr[kj], tpr[kj]), xytext=(0.46, 0.06),
                textcoords="axes fraction", fontsize=6.5, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
    mf = cal["fixed_fpr"]["metrics"]
    tp_fpr = mf["tp"] / (mf["tp"] + mf["fn"]) if (mf["tp"] + mf["fn"]) else 0.0
    ax.axvline(fpr_target, color=GREY, ls=":", lw=0.9)
    ax.scatter([fpr_target], [tp_fpr], s=30, color=GREY, zorder=5, marker="o",
               edgecolor="white", linewidth=0.6)
    ax.text(fpr_target, tp_fpr + 0.06, f"FPR≤{fpr_target:.0%}", ha="right", va="bottom",
            fontsize=6, color=GREY)
    m05 = metrics_at(gt, score, 0.5)
    tpr05 = m05["tp"] / (m05["tp"] + m05["fn"]) if (m05["tp"] + m05["fn"]) else 0.0
    fpr05 = m05["fp"] / (m05["fp"] + m05["tn"]) if (m05["fp"] + m05["tn"]) else 0.0
    ax.scatter([fpr05], [tpr05], s=30, color=RED, zorder=5, edgecolor="white", linewidth=0.6)
    ax.text(fpr05, tpr05 - 0.09, "0.5", ha="right", va="top", fontsize=6, color=RED)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("ROC — method 2: Youden J", loc="left", pad=3)
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    for a, t in zip(axes, "abc"):
        panel_label(a, t)

    fig.suptitle("Alarm-threshold calibration on the validation set (TCN DL branch, wandering)",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    return fig


# --------------------------- 主流程 ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_traj", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr", type=float, default=0.05)
    args = ap.parse_args()

    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma, offset = stats["mu"], stats["sigma"], stats.get("offset", 2.0)
    print(f"model obs={obs} pred={pred}  mu={mu:.2f}m sigma={sigma:.2f}m offset={offset}")

    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, args.max_traj, args.seed)
    print(f"评估轨迹数 = {len(trajs)}")

    out = {"meta": {"behavior": "wandering", "max_traj": args.max_traj, "seed": args.seed,
                    "score": "S_dl (ade_to_score, [0,1])", "mu": mu, "sigma": sigma,
                    "offset": offset, "fpr_target": args.fpr}}

    for name, inj in (("wandering", T.inject_loiter), ("loitering", T.inject_stop)):
        ade, gt = labeled_ade(model, trajs, inj, obs, pred, stride=2)
        s_dl = T.ade_to_score(ade, mu, sigma, offset)
        s_dl = np.nan_to_num(s_dl, nan=0.0)
        auroc = float(metrics.roc_auc_score(gt, s_dl)) if 0 < gt.sum() < len(gt) else None
        print(f"\n[{name}]  n_window={len(gt)} (pos={int(gt.sum())}, neg={int((1-gt).sum())})  "
              f"AUROC={auroc:.3f}")
        if name == "wandering":
            cal = calibrate(gt, s_dl, args.fpr)
            out["data"] = {"scores": [round(float(v), 4) for v in s_dl],
                           "labels": [int(v) for v in gt]}
            out["calibration"] = cal
            for meth in ("f1_optimal", "youden_j", "fixed_fpr"):
                t = cal[meth]["threshold"]; m = cal[meth]["metrics"]
                print(f"  {meth:<12} tau={t}  P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f} acc={m['accuracy']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
            m05 = metrics_at(gt, s_dl, 0.5)
            out["at_0_5"] = m05
            print(f"  {'current 0.5':<12} tau=0.5     P={m05['precision']:.3f} R={m05['recall']:.3f} "
                  f"F1={m05['f1']:.3f} acc={m05['accuracy']:.3f} (TP={m05['tp']} FP={m05['fp']} FN={m05['fn']})")
        else:
            out["loitering"] = {"auroc": round(auroc, 4) if auroc is not None else None,
                                "note": "AUROC<0.5 structural blind spot — no usable threshold"}

    json.dump(out, open(os.path.join(_HERE, "threshold_calib.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nsaved threshold_calib.json")

    fig = fig_calibration(np.array(out["data"]["labels"]),
                          np.array(out["data"]["scores"]),
                          out["calibration"], args.fpr)
    save(fig, "fig_threshold_calib.png", outdir=os.path.join(_HERE, "figs"))


if __name__ == "__main__":
    main()
