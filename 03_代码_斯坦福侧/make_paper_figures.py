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
"""
论文配图（方法示意图，英文，与实验图同风格）
============================================

生成 results/paper_figs/ 下的方法类图：
  fig_framework.png         整体框架：EVT/规则分支 + TCN 分支 + 置信门限融合
  fig_fluxev_stages.png     FluxEV 两步平滑各阶段(速度通道注入疾跑)
  fig_anomaly_types.png     四种注入异常轨迹示意
  fig_tcn_prediction.png    TCN 轨迹预测示例(观测→预测→真值)
  fig_ade_distribution.png  正常 vs 异常窗口 ADE 分布
  fig_training_curve.png    TCN 训练收敛曲线(Geolife 满规模)
"""
import os
import json

import sdd_common as S
import tcn_branch as T

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS = os.path.join(S.RESULTS_DIR, "paper_figs")
os.makedirs(RESULTS, exist_ok=True)

C_INK = S.C_INK
C_MUTED = S.C_MUTED
C_GRID = S.C_GRID
C_RULE = S.C_RULE
C_DL = S.C_DL
C_FLUXEV = S.C_FLUXEV
C_GT = S.C_GT


def _save(fig, name):
    fig.savefig(os.path.join(RESULTS, name), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  +", os.path.join(RESULTS, name))


# ============================ 图1：整体框架 ============================

def fig_framework():
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.set_xlim(0, 100); ax.set_ylim(2, 63); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=8.4, tc="white", bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.3",
                           fc=fc, ec=ec, lw=1.0)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, weight="bold" if bold else "normal", linespacing=1.35)
        return (x, y, w, h)

    def arrow(x1, y1, x2, y2, color="0.35"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color=color))

    # 输入
    box(30, 56, 40, 5.4, "Tracked pedestrian trajectory  $(x_t, y_t)$  (m, $\\Delta t$)",
        "#f2f2ef", "0.55", tc=C_INK, bold=True)

    # 分支容器
    box(3, 30, 44, 21.5, "", "#eef3ee", "0.4")
    box(53, 30, 44, 21.5, "", "#fdf1ec", "0.4")

    # Branch 1
    ax.text(25, 49.3, "Branch 1 — EVT + hard rules", ha="center", fontsize=9.4,
            color="#1B7837", weight="bold")
    box(6, 40.5, 18, 6, "FluxEV-SPOT (EVT)\n$(v,a,\\theta)$ $\\to$ 2-step\nsmooth $\\to$ GPD/MOM",
        "#1B7837", "#1B7837", fs=7.3)
    box(27, 40.5, 17, 6, "Hard if-rules\ndwell / gather /\ngeofence", "#1B7837", "#1B7837", fs=7.3)
    box(6, 33.5, 18, 4, "$S^{\\mathrm{motion}}$", "#d9e9d9", "#1B7837", tc=C_INK)
    box(27, 33.5, 17, 4, "$S^{\\mathrm{rule}}$", "#d9e9d9", "#1B7837", tc=C_INK)
    box(9, 24.5, 32, 5.2, "$S^{\\mathrm{full}}=\\max(S^{\\mathrm{motion}},S^{\\mathrm{rule}})$",
        "#1B7837", "#1B7837", fs=8.6, bold=True)

    # Branch 2
    ax.text(75, 49.3, "Branch 2 — deep learning (TCN)", ha="center", fontsize=9.4,
            color="#c25a1e", weight="bold")
    box(56, 40.5, 18, 6, "TCN predictor\npast 16 frames\n$\\to$ next 8 frames",
        "#eb6834", "#eb6834", fs=7.3)
    box(77, 40.5, 17, 6, "Calibration\nsigmoid / z-clip", "#eb6834", "#eb6834", fs=7.3)
    box(56, 33.5, 18, 4, "ADE$_t$", "#f8d9c6", "#eb6834", tc=C_INK)
    box(77, 33.5, 17, 4, "$S^{\\mathrm{dl}}$", "#f8d9c6", "#eb6834", tc=C_INK)
    box(60, 24.5, 32, 5.2, "confidence gate\n$S^{\\mathrm{dl}}\\!\\cdot\\!\\mathbf{1}[S^{\\mathrm{dl}}\\!\\geq\\!\\tau_g]$",
        "#eb6834", "#eb6834", fs=7.6, bold=True)

    # 输入分叉
    arrow(40, 55.6, 25, 49.8)
    arrow(60, 55.6, 75, 49.8)

    # Branch 1 内部
    arrow(15, 40.1, 15, 38.0)
    arrow(35.5, 40.1, 35.5, 38.0)
    arrow(15, 33.1, 19, 30.2)
    arrow(35.5, 33.1, 31, 30.2)

    # Branch 2 内部
    arrow(65, 40.1, 65, 38.0)
    arrow(85.5, 40.1, 85.5, 38.0)
    arrow(65, 33.1, 69, 30.2)
    arrow(85.5, 33.1, 81, 30.2)

    # 融合 + 告警
    box(28, 14.5, 44, 6, "Fusion  $S = \\max(S^{\\mathrm{full}},\\ S^{\\mathrm{dl}}\\!\\cdot\\!\\mathbf{1}[S^{\\mathrm{dl}}\\!\\geq\\!\\tau_g])$",
        C_INK, C_INK, fs=8.8, bold=True)
    box(38, 5.5, 24, 5, "Alarm  $\\hat{y} = \\mathbf{1}[S \\geq 0.5]$", "#5a5a5a", "#5a5a5a",
        fs=8.6, bold=True)

    arrow(25, 24.1, 46, 21.0)
    arrow(75, 24.1, 54, 21.0)
    arrow(50, 14.1, 50, 11.0)

    ax.set_title("Overall framework: EVT/rule branch + TCN branch with confidence-gated fusion",
                 fontsize=10, loc="left")
    _save(fig, "fig_framework.png")


# ============================ 图2：FluxEV 两阶段平滑 ============================

def fig_fluxev_stages():
    trajs = S.collect_sdd_walking(1.0, 120, seed=3, min_len=80)
    sc, x, y, t = max(trajs, key=lambda z: len(z[1]))
    n = len(x)
    # 注入疾跑(速度通道突变)
    xa, ya = x.copy(), y.copy()
    i0, i1 = int(n * 0.55), min(int(n * 0.55) + max(16, int(n * 0.18)), n - 2)
    T.inject_run(xa, ya, i0, i1)

    speed, accel, turn = S.run_sdd.features_2d(xa, ya, 1.0)
    train_len = n // 2
    period = max(2, min(30, max(2, n // 8)))
    s_w = max(2, min(10, max(2, n // 30)))
    sc_speed, ss_err, ths, stages = S.gp.fluxev_score(speed, train_len, period,
                                                       s_w=s_w, p_w=3, half_d_w=1, q=0.005)

    fig, axes = plt.subplots(5, 1, figsize=(9.5, 9.5), sharex=True)
    tt = t
    axes[0].plot(tt, speed, color="#2a78d6", lw=1.1)
    axes[0].axvspan(t[i0], t[i1], color=C_GT, alpha=0.12, lw=0)
    axes[0].set_ylabel("speed (m/s)")
    axes[0].set_title("FluxEV two-step smoothing (speed channel, injected sprint)",
                      loc="left", fontsize=9.5)

    axes[1].plot(tt, stages["pred_err"], color="#898781", lw=0.9)
    axes[1].set_ylabel("residual $e_i$")
    axes[2].plot(tt, stages["fs_err"], color="#eb6834", lw=0.9)
    axes[2].set_ylabel("1st smooth $f_i$")
    axes[3].plot(tt, stages["ss_err"], color="#1B7837", lw=1.1)
    axes[3].set_ylabel("2nd smooth $g_i$")

    ax = axes[4]
    ax.plot(t[train_len:], sc_speed, color="#e34948", lw=1.3)
    ax.axhline(0.5, color=C_INK, ls="--", lw=0.9)
    ax.axvspan(t[i0], t[i1], color=C_GT, alpha=0.12, lw=0)
    ax.set_ylabel("$S^{\\mathrm{ev}}$")
    ax.set_xlabel("time (s)")
    for a in axes:
        a.grid(True, color=C_GRID, lw=0.5); a.set_axisbelow(True)
        a.axvspan(t[i0], t[i1], color=C_GT, alpha=0.06, lw=0)
    fig.tight_layout()
    _save(fig, "fig_fluxev_stages.png")


# ============================ 图3：异常注入类型 ============================

def fig_anomaly_types():
    trajs = S.collect_sdd_walking(1.0, 120, seed=5, min_len=80)
    sc, x, y, t = max(trajs, key=lambda z: len(z[1]))
    kinds = [("Wandering", T.inject_loiter, C_WANDER := "#2a78d6"),
             ("Loitering-in-place", T.inject_stop, "#eb6834"),
             ("Sprint", T.inject_run, "#eda100"),
             ("Zigzag", T.inject_zigzag, "#1baf7a")]
    n = len(x)
    i0, i1 = int(n * 0.5), min(int(n * 0.75), n - 2)

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.1))
    for ax, (name, inj, color) in zip(axes, kinds):
        xa, ya = x.copy(), y.copy()
        inj(xa, ya, i0, i1)
        ax.plot(xa[:i0], ya[:i0], lw=0.8, color="0.75", label="normal")
        ax.plot(xa[i0:i1], ya[i0:i1], lw=1.6, color=color, label="anomaly")
        ax.plot(xa[i1:], ya[i1:], lw=0.8, color="0.75")
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, color=C_GRID, lw=0.4)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Injected motion-pattern anomalies (Stanford Drone trajectory)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "fig_anomaly_types.png")


# ============================ 图4：TCN 预测示例 ============================

def fig_tcn_prediction():
    model, stats, cfg = S.ensure_sdd_tcn()
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    trajs = S.collect_sdd_walking(1.0, 120, seed=11, min_len=60)
    sc, x, y, t = max(trajs, key=lambda z: len(z[1]))
    s = int(len(x) * 0.45)
    ref = np.array([x[s + obs - 1], y[s + obs - 1]])
    x_obs = np.stack([x[s:s + obs], y[s:s + obs]], -1) - ref
    y_true = np.stack([x[s + obs:s + obs + pred], y[s + obs:s + obs + pred]], -1) - ref
    y_hat = T.predict(model, x_obs)
    # 拼接连续轨迹(观测尾部 + 预测/真值)
    obs_abs = np.stack([x[s:s + obs], y[s:s + obs]], -1)
    pred_abs = np.concatenate([obs_abs[-1:], obs_abs[-1:] + y_hat], 0)
    true_abs = np.concatenate([obs_abs[-1:], obs_abs[-1:] + y_true], 0)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.plot(obs_abs[:, 0], obs_abs[:, 1], color="0.6", lw=2.0, label="observed (16 frames)")
    ax.plot(pred_abs[:, 0], pred_abs[:, 1], color=C_DL, lw=2.0, ls="--",
            label="predicted (8 frames)")
    ax.plot(true_abs[:, 0], true_abs[:, 1], color=C_GT, lw=2.0,
            label="actual (8 frames)")
    ax.scatter(obs_abs[-1, 0], obs_abs[-1, 1], s=40, color=C_INK, zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("TCN trajectory prediction (x → y)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color=C_GRID, lw=0.5)
    fig.tight_layout()
    _save(fig, "fig_tcn_prediction.png")


# ============================ 图5：ADE 分布 ============================

def fig_ade_distribution():
    model, stats, cfg = S.ensure_sdd_tcn()
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    trajs = S.collect_sdd_walking(1.0, 150, seed=21, min_len=48)
    total = obs + pred
    normal, anomal = [], []
    for sc, x, y, t in trajs:
        n = len(x)
        if n < total + 24:
            continue
        i0, i1 = int(n * 0.55), min(int(n * 0.55) + max(16, int(n * 0.18)), n - total - 2)
        if i1 - i0 < 8:
            continue
        xa, ya = x.copy(), y.copy()
        mask = T.inject_loiter(xa, ya, i0, i1)
        obs_l, fut_l = [], []
        for s in range(0, n - total + 1, 2):
            ref = np.array([xa[s + obs - 1], ya[s + obs - 1]])
            obs_l.append(np.stack([xa[s:s + obs], ya[s:s + obs]], -1) - ref)
            fut_l.append(np.stack([xa[s + obs:s + total], ya[s + obs:s + total]], -1) - ref)
        ade = T.compute_ade(model, np.asarray(obs_l, np.float32),
                            np.asarray(fut_l, np.float32))
        for s, a in zip(range(0, n - total + 1, 2), ade):
            fut = np.arange(s + obs, s + total)
            (anomal if np.isin(fut, mask).any() else normal).append(a)

    normal = np.array(normal); anomal = np.array(anomal)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, max(normal.max(), anomal.max()) * 0.98, 40)
    ax.hist(normal, bins, alpha=0.55, color="#2a78d6", label="normal windows",
            density=True)
    ax.hist(anomal, bins, alpha=0.55, color=C_GT, label="anomalous (wandering)",
            density=True)
    ax.axvline(stats["mu"], color=C_INK, ls="--", lw=1.0, label="calibrated $\\mu$")
    ax.set_xlabel("ADE (m)"); ax.set_ylabel("density")
    ax.set_title("ADE distribution: normal vs anomalous (wandering)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color=C_GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_ade_distribution.png")


# ============================ 图6：训练曲线 ============================

def fig_training_curve():
    hist_path = os.path.join(_PKGP.TCN_DIR, "results", "train_history.json")
    if not os.path.exists(hist_path):
        print("  [skip] 未找到", hist_path)
        return
    hist = json.load(open(hist_path, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(hist["train"], color="#2a78d6", lw=1.4, label="train loss (MSE)")
    ax.plot(hist["val"], color="#eb6834", lw=1.4, label="val loss (MSE)")
    ax.axvline(hist.get("best_epoch", 0) - 1, color=C_MUTED, ls=":", lw=1.0)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (m$^2$)")
    ax.set_title("TCN training convergence (Geolife, full scale)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color=C_GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_training_curve.png")


if __name__ == "__main__":
    fig_framework()
    fig_fluxev_stages()
    fig_anomaly_types()
    fig_tcn_prediction()
    fig_ade_distribution()
    fig_training_curve()
    print("完成，输出在", RESULTS)
