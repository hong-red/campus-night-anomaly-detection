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
"""原地逗留(loitering)的报警阈值定标 —— 规则分支 dwell，不是 DL 分支。

背景：DL(TCN) 对逗留是结构盲区（AUROC 0.19），但逗留由**规则分支 dwell**负责，
      `stationary_episode_score(speed, dt, lo=120, hi=300)`。这里对 dwell 分数做
      和 DL 分支一样的三种定标，并额外扫 `lo`（这才是逗留真正的杠杆）。

两种正交的旋钮，务必分开看：
  τ（报警阈值）：dwell 分数是「0 / 线性爬坡」，负例(正常行走)分数恒 0 → FP=0，
                所以 τ 只要够低就能接住爬坡段；F1 最优是 τ→0+，但被因果延迟封顶。
  lo（爬坡起点）：dwell 唯一短板是 120s 才起步，漏掉每段逗留前 120s。
                降 lo 才是提升召回、且不引入 FP 的正路（md 校准②「提升」）。

输出：
  dwell_threshold_calib.json   τ 扫描 + lo 扫描 + 三种阈值
  figs/fig_dwell_calib.png     (a) F1/P/R vs τ   (b) 召回 vs lo（FP=0）
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.CODE_TSINGHUA)
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics

import geolife_pipeline as gp
from style import apply_nature_style, C, DOUBLE_COL, MM, panel_label, save

apply_nature_style()

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0
DWELL_SEC = 360
HERE = os.path.dirname(os.path.abspath(__file__))

GREEN = C["green"]   # dwell 规则
BLUE = C["calm"]     # F1 曲线
ORANGE = C["accent"] # 召回
GREY = C["grey"]
RED = C["alert"]


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


def tau_sweep(gt, score, n=400):
    """在 (0,1] 上扫 τ，返回 P/R/F1/acc。避开 τ=0（含全部负例）。"""
    ts = np.linspace(1e-3, 1.0, n)
    pos_score, neg_score = score[gt == 1], score[gt == 0]
    tp = np.array([(pos_score >= t).sum() for t in ts], dtype=float)
    fp = np.array([(neg_score >= t).sum() for t in ts], dtype=float)
    fn = gt.sum() - tp
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    acc = (tp + (len(neg_score) - fp)) / len(gt)
    return ts, prec, rec, f1, acc


def calibrate(gt, score, fpr_target=0.05):
    auroc = float(metrics.roc_auc_score(gt, score)) if 0 < gt.sum() < len(gt) and np.std(score) > 1e-9 else None
    ap = float(metrics.average_precision_score(gt, score)) if auroc is not None else None

    prec, rec, thr = metrics.precision_recall_curve(gt, score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.argmax(f1))
    tau_f1 = float(thr[k]) if k < len(thr) else 0.0

    fpr, tpr, thr_roc = metrics.roc_curve(gt, score)
    j = tpr - fpr
    kj = int(np.argmax(j))
    tau_j = float(thr_roc[kj])
    ok = fpr <= fpr_target
    idx = int(np.where(ok)[0][-1]) if ok.any() else 0
    tau_fpr = float(thr_roc[idx])

    return {
        "auroc": round(auroc, 4) if auroc is not None else None,
        "ap": round(ap, 4) if ap is not None else None,
        "f1_optimal": {"threshold": round(tau_f1, 4), "metrics": metrics_at(gt, score, tau_f1)},
        "youden_j":   {"threshold": round(tau_j, 4), "metrics": metrics_at(gt, score, tau_j)},
        "fixed_fpr":  {"threshold": round(tau_fpr, 4), "fpr_target": fpr_target,
                       "metrics": metrics_at(gt, score, tau_fpr)},
    }


def main():
    print("扫描长行走轨迹 (min_dur=900s, max_files=2000) ...")
    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    print(f"长轨迹 = {len(walks)} 条")

    pool_gt = []
    pool = {lo: [] for lo in (120, 60, 30, 15)}

    for (npts, med, path, lat5, lon5, tg5) in walks:
        lat, lon, tg = gp.resample(lat5, lon5, tg5, DT)
        n = len(lat)
        if n < int(700 / DT):
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + int(DWELL_SEC / DT), n - 2)
        lat = lat.copy(); lon = lon.copy()
        lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
        gt = np.zeros(n); gt[i0:i1] = 1

        speed, _, _ = gp.trajectory_features(lat, lon, DT)
        mask = np.arange(n) >= n // 2
        g = gt[mask]
        if g.sum() == 0 or g.sum() == len(g):
            continue
        pool_gt.append(g)
        for lo in pool:
            pool[lo].append(gp.stationary_episode_score(speed, DT, lo, 300)[mask])

    g_all = np.concatenate(pool_gt)
    print(f"逐帧池化：{len(g_all)} 帧，正例 {int(g_all.sum())}，负例 {int((1 - g_all).sum())}")

    out = {"meta": {"behavior": "loitering", "detector": "dwell rule (stationary_episode_score)",
                    "dt": DT, "inject_dwell_s": DWELL_SEC, "hi": 300, "note":
                    "负例(正常行走)分数恒0 → FP=0；τ只需够低接住爬坡段，真正杠杆是 lo"}}

    # ---- 基线 dwell(120/300) 的 τ 定标 ----
    s120 = np.concatenate(pool[120])
    out["dwell_120_300"] = {"tau_sweep": {}, "calibration": calibrate(g_all, s120)}
    ts, prec, rec, f1, acc = tau_sweep(g_all, s120)
    out["dwell_120_300"]["tau_sweep"] = {
        "tau": [round(float(t), 4) for t in ts],
        "precision": [round(float(v), 4) for v in prec],
        "recall": [round(float(v), 4) for v in rec],
        "f1": [round(float(v), 4) for v in f1],
        "accuracy": [round(float(v), 4) for v in acc],
    }

    # ---- lo 扫描（固定低阈值 τ=0.02，只看召回与 FP） ----
    print("\nlo 扫描 @τ=0.02（FP 应恒 0）：")
    lo_rows = {}
    for lo in (120, 60, 30, 15):
        s = np.concatenate(pool[lo])
        m = metrics_at(g_all, s, 0.02)
        auroc = float(metrics.roc_auc_score(g_all, s))
        lo_rows[str(lo)] = {**m, "auroc": round(auroc, 4)}
        print(f"  lo={lo:>3}s  rec={m['recall']:.3f}  F1={m['f1']:.3f}  acc={m['accuracy']:.3f}  "
              f"FP={m['fp']}  AUROC={auroc:.3f}")
    out["lo_sweep_at_tau_0_02"] = lo_rows

    # ---- 三种阈值汇报 ----
    cal = out["dwell_120_300"]["calibration"]
    print("\n三种定标阈值（dwell 120/300 分数）：")
    for meth in ("f1_optimal", "youden_j", "fixed_fpr"):
        t = cal[meth]["threshold"]; m = cal[meth]["metrics"]
        print(f"  {meth:<12} tau={t:<6} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} acc={m['accuracy']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    json.dump(out, open(os.path.join(HERE, "dwell_threshold_calib.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nsaved dwell_threshold_calib.json")

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 52 * MM))

    # (a) F1 / P / R / acc vs τ
    ax = axes[0]
    ts, prec, rec, f1, acc = (np.array(out["dwell_120_300"]["tau_sweep"]["tau"]),
                              np.array(out["dwell_120_300"]["tau_sweep"]["precision"]),
                              np.array(out["dwell_120_300"]["tau_sweep"]["recall"]),
                              np.array(out["dwell_120_300"]["tau_sweep"]["f1"]),
                              np.array(out["dwell_120_300"]["tau_sweep"]["accuracy"]))
    ax.plot(ts, f1, color=BLUE, lw=1.6, label="F1")
    ax.plot(ts, rec, color=ORANGE, lw=1.2, ls="--", label="Recall")
    ax.plot(ts, prec, color=GREEN, lw=1.2, ls=":", label="Precision (=1, FP=0)")
    kf = int(np.argmax(f1))
    ax.scatter([ts[kf]], [f1[kf]], s=40, color=BLUE, zorder=5, marker="*",
               edgecolor="white", linewidth=0.6)
    ax.axvline(0.5, color=RED, ls="--", lw=1.0)
    ax.text(0.5, 0.10, "0.5", ha="left", va="bottom", fontsize=6, color=RED)
    ax.annotate(f"max F1={f1[kf]:.2f} @ $\\tau$={ts[kf]:.3f}\n"
                f"(recall ceiling {rec[kf]:.2f} = causal delay)",
                (ts[kf], f1[kf]), xytext=(0.28, 0.32), textcoords="axes fraction",
                fontsize=6.5, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    ax.set_xlabel("alarm threshold $\\tau$ (dwell score)")
    ax.set_ylabel("score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("dwell rule — F1/P/R vs $\\tau$", loc="left", pad=3)
    ax.legend(frameon=False, fontsize=6, loc="center right")

    # (b) 召回 vs lo（真正的杠杆）
    ax = axes[1]
    los = [120, 60, 30, 15]
    recs = [lo_rows[str(lo)]["recall"] for lo in los]
    f1s = [lo_rows[str(lo)]["f1"] for lo in los]
    x = np.arange(len(los))
    w = 0.34
    ax.bar(x - w/2, recs, w, color=ORANGE, label="Recall @τ=0.02", zorder=3)
    ax.bar(x + w/2, f1s, w, color=BLUE, label="F1 @τ=0.02", zorder=3)
    for i in range(len(los)):
        ax.text(i - w/2, recs[i] + 0.015, f"{recs[i]:.2f}", ha="center", fontsize=6.5)
        ax.text(i + w/2, f1s[i] + 0.015, f"{f1s[i]:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([f"lo={lo}s" for lo in los], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_xlabel("dwell ramp start $lo$  (FP = 0 throughout)")
    ax.set_title("lowering $lo$ lifts recall — the real lever", loc="left", pad=3)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)

    for a, t in zip(axes, "ab"):
        panel_label(a, t)

    fig.suptitle("Loitering (原地逗留) calibration — rule branch, not the DL blind spot",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig_dwell_calib.png", outdir=os.path.join(HERE, "figs"))


if __name__ == "__main__":
    main()
