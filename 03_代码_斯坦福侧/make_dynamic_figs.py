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
动态图(GIF)：把 TCN 检测过程与感受野"动起来"
=============================================

输出 results/paper_figs/：
  anim_tcn_prediction.gif    滑动窗口预测 + S_dl 异常分数实时上升(真实轨迹注入徘徊)
  anim_receptive_field.gif   扩张因果卷积感受野自顶向下逐层扩张
"""
import os
import tempfile

import sdd_common as S
import tcn_branch as T

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RESULTS = os.path.join(S.RESULTS_DIR, "paper_figs")
os.makedirs(RESULTS, exist_ok=True)

C_TCN = "#2a78d6"
C_DL = "#eb6834"
C_GT = "#e34948"
C_INK = "#0b0b0b"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"


def _to_gif(frames_dir, out_path, duration=110):
    files = sorted(os.listdir(frames_dir))
    imgs = [Image.open(os.path.join(frames_dir, f)).convert("RGB") for f in files]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration,
                 loop=0, optimize=True)
    print("  +", out_path, f"({len(imgs)} frames)")


# ============================ 1) TCN 预测滑动 + 异常分数 ============================

def anim_tcn_prediction():
    model, stats, cfg = S.ensure_sdd_tcn()
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    trajs = S.collect_sdd_walking(1.0, 120, seed=13, min_len=70)
    sc, x, y, t = max(trajs, key=lambda z: len(z[1]))
    n = len(x)
    i0 = int(n * 0.55)
    i1 = min(i0 + max(16, int(n * 0.18)), n - obs - pred - 2)
    xa, ya = x.copy(), y.copy()
    T.inject_loiter(xa, ya, i0, i1)

    ade = S.frame_ade(model, xa, ya, obs, pred)
    s_dl = np.nan_to_num(T.ade_to_score(ade, stats["mu"], stats["sigma"],
                                        stats.get("offset", 2.0)), nan=0.0)

    steps = np.linspace(obs, n - obs - pred, 90, dtype=int)
    tmp = tempfile.mkdtemp()
    for k, s in enumerate(steps):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
        ax = axes[0]
        ax.plot(xa, ya, color="0.82", lw=0.8, zorder=1)
        ax.scatter(xa[i0:i1], ya[i0:i1], s=4, color=C_GT, alpha=0.5, zorder=2,
                   label="injected anomaly")
        ax.plot(xa[s:s + obs], ya[s:s + obs], color=C_TCN, lw=2.4, zorder=4,
                label="observed (16)")
        ref = np.array([xa[s + obs - 1], ya[s + obs - 1]])
        xo = np.stack([xa[s:s + obs], ya[s:s + obs]], -1) - ref
        yhat = T.predict(model, xo)
        pred_abs = ref + yhat
        ax.plot(np.r_[xa[s + obs - 1], pred_abs[:, 0]],
                np.r_[ya[s + obs - 1], pred_abs[:, 1]], color=C_DL, ls="--", lw=2.2,
                zorder=5, label="predicted (8)")
        ax.plot(xa[s + obs - 1:s + obs + pred], ya[s + obs - 1:s + obs + pred],
                color=C_GT, lw=2.0, zorder=6, label="actual (8)")
        ax.scatter(xa[s + obs - 1], ya[s + obs - 1], s=30, color=C_INK, zorder=7)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"TCN sliding prediction (t={t[s + obs]:.0f}s)", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=6.8, loc="lower right")
        ax.grid(True, color=C_GRID, lw=0.4)
        ax.set_aspect("equal", adjustable="datalim")

        ax = axes[1]
        ax.plot(t, s_dl, color=C_DL, lw=1.2)
        ax.fill_between(t, 0, 1, where=(np.arange(n) >= i0) & (np.arange(n) < i1),
                        color=C_GT, alpha=0.12, lw=0)
        ax.axvline(t[s + obs], color=C_INK, lw=1.3)
        ax.axhline(0.5, color=C_INK, ls="--", lw=0.9)
        ax.set_xlabel("time (s)"); ax.set_ylabel("$S^{\\mathrm{dl}}$")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title("anomaly score $S^{\\mathrm{dl}}(t)$", loc="left", fontsize=9)
        ax.grid(True, color=C_GRID, lw=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(tmp, f"f{k:03d}.png"), dpi=110)
        plt.close(fig)

    _to_gif(tmp, os.path.join(RESULTS, "anim_tcn_prediction.gif"))


# ============================ 2) 感受野逐层扩张 ============================

def anim_receptive_field():
    dilations = [1, 2, 4, 8]
    Tlen = 16
    output = (4, Tlen - 1)
    edges, reachable = [], {output}

    def expand(node):
        l, tt = node
        if l == 0:
            return
        d = dilations[l - 1]
        for j in range(3):
            t2 = tt - j * d
            if t2 >= 0:
                child = (l - 1, t2)
                edges.append((child, node))
                if child not in reachable:
                    reachable.add(child)
                    expand(child)

    expand(output)
    field_inputs = {tt for (l, tt) in reachable if l == 0}
    layers = {l: [(ll, tt) for (ll, tt) in reachable if ll == l] for l in range(5)}
    # 自顶向下逐层显示：层 4 -> 0
    reveal = list(range(4, -1, -1))

    tmp = tempfile.mkdtemp()
    for k in range(len(reveal) + 6):
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        y_of = lambda l: l + 0.3
        visible = set(reveal[:min(k, len(reveal))])
        # 边
        for (child, parent) in edges:
            (cl, ct), (pl, pt) = child, parent
            if pl in visible and cl in visible:
                ax.plot([ct, pt], [y_of(cl), y_of(pl)], color="#9ec5f4", lw=0.8, zorder=1)
        # 节点
        for l in range(5):
            if l not in visible:
                continue
            for (ll, tt) in layers[l]:
                if l == 0:
                    in_field = tt in field_inputs
                    ax.scatter(tt, y_of(l), s=90,
                               color=C_TCN if in_field else "#d5d5d0", zorder=3,
                               edgecolor="white", linewidth=0.6)
                else:
                    ax.scatter(tt, y_of(l), s=45, color="#6aa4e8", zorder=3,
                               edgecolor="white", linewidth=0.5)
        # 输出节点(最后帧高亮脉冲)
        if 4 in visible:
            pulse = 120 if k >= len(reveal) else 90
            ax.scatter(Tlen - 1, y_of(4), s=pulse, color=C_GT, zorder=4,
                       edgecolor="white", linewidth=0.8)
            ax.annotate("output", (Tlen - 1, y_of(4)), textcoords="offset points",
                        xytext=(8, 4), fontsize=8, color=C_GT, weight="bold")
        # 输入底(未揭示时灰)
        for tt in range(Tlen):
            if 0 not in visible or (0, tt) not in reachable:
                ax.scatter(tt, y_of(0), s=90, color="#e8e8e4", zorder=2,
                           edgecolor="white", linewidth=0.6)
        for l, d in enumerate(dilations):
            ax.text(-1.6, y_of(l + 1), f"d={d}", fontsize=8, color=C_MUTED, ha="right",
                    va="center")
        ax.text(-1.6, y_of(0), "input", fontsize=8, color=C_MUTED, ha="right", va="center")
        ax.set_xlim(-3.5, Tlen + 0.5); ax.set_ylim(-0.5, 5.4)
        ax.set_xticks(range(Tlen)); ax.set_xlabel("time step (causal, left→right)")
        ax.set_yticks([])
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color=C_GRID, lw=0.5)
        ax.set_title("Receptive field expansion (top-down): d=8 → 4 → 2 → 1",
                     fontsize=9.5, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(tmp, f"f{k:03d}.png"), dpi=110)
        plt.close(fig)

    _to_gif(tmp, os.path.join(RESULTS, "anim_receptive_field.gif"), duration=700)


if __name__ == "__main__":
    anim_tcn_prediction()
    anim_receptive_field()
    print("完成，输出在", RESULTS)
