#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多人聚集仿真 —— 纯合成（不依赖 Geolife）
========================================
生成一个主行人 + 若干其他行人，注入一段「多人聚集」事件，用规则检测：
    n_near[i] = 最近 180s 内进入 50m 半径的「他人」数量
    score[i]  = clip(n_near / (GATHER_MIN_PEOPLE - 1), 0, 1)   # -1 = 主行人自己算 1 人
    alarm[i]  = n_near >= GATHER_MIN_PEOPLE - 1                 # 凑满 >=3 人

输出 results/simulation.png 与 results/metrics.csv。

判定与 geolife_pipeline.gather_score 完全一致（这里用平面坐标，米）。

用法：python simulate.py [--n 2400] [--dt 0.5] [--seed 1]
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

warnings.filterwarnings("ignore")

# --------------------------- 常量（与 geolife_pipeline 对齐） ---------------------------
GATHER_RADIUS = 50      # m，聚集空间半径
GATHER_DT = 180         # s，聚集时间窗
GATHER_MIN_PEOPLE = 3   # 深夜 >=3 人


# --------------------------- 聚集规则 ---------------------------

def gather_score(main_x, main_y, others, dt, radius=GATHER_RADIUS,
                 window=GATHER_DT):
    """
    多人聚集：最近 window 秒内任一时刻进入 radius 米的「他人」数量 -> 分数。
    others: list[(x[], y[])]，与主行人同一时间网格(dt)。
    返回 (score[0,1], n_near)。score = clip(n_near / (MIN_PEOPLE-1), 0, 1)。
    """
    n = len(main_x)
    n_near = np.zeros(n, dtype=int)
    if not others:
        return np.zeros(n), n_near
    w = max(1, int(window / dt))          # 时间窗换算成步数
    for ox, oy in others:
        for i in range(n):
            lo = max(0, i - w)
            d = np.hypot(main_x[i] - ox[lo:i + 1], main_y[i] - oy[lo:i + 1])
            if np.min(d) <= radius:
                n_near[i] += 1
    score = np.clip(n_near / (GATHER_MIN_PEOPLE - 1), 0.0, 1.0)
    return score, n_near


# --------------------------- 场景生成 ---------------------------

def build_scene(n, dt, rng):
    """主行人沿 x 轴匀速走；2 个「聚集者」在 [t0,t1] 贴到 50m 内，1 个「路人」始终 60m 外。"""
    v = 1.4
    main_x = v * np.arange(n) * dt
    main_y = np.zeros(n)
    t = np.arange(n) * dt

    t0 = int(0.55 * n)     # 聚集开始
    t1 = int(0.75 * n)     # 聚集结束

    others = []
    # 2 个聚集者（主行人自己 + 2 = 3 人）
    for k, (dx, dy) in enumerate([(12.0, 0.0), (0.0, 12.0)]):
        ox = np.full(n, 120.0 + k * 5.0)
        oy = np.full(n, 120.0 + k * 5.0)
        ox[t0:t1] = main_x[t0:t1] + dx
        oy[t0:t1] = main_y[t0:t1] + dy
        others.append((ox, oy))
    # 1 个路人：始终 60m 外（验证不误报）
    ox = main_x + 60.0
    oy = np.full(n, 60.0)
    others.append((ox, oy))

    return main_x, main_y, others, t, t0, t1


# --------------------------- 评估 ---------------------------

def pointwise_metrics(pred, gt):
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser(description="多人聚集规则仿真")
    ap.add_argument("--n", type=int, default=2400, help="总时间步数")
    ap.add_argument("--dt", type=float, default=0.5, help="采样间隔(s)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, dt = args.n, args.dt

    main_x, main_y, others, t, t0, t1 = build_scene(n, dt, rng)
    score, n_near = gather_score(main_x, main_y, others, dt)
    alarm = (n_near >= GATHER_MIN_PEOPLE - 1).astype(int)   # >=3 人

    gt = np.zeros(n, dtype=int)
    gt[t0:t1] = 1

    p, r, f1 = pointwise_metrics(alarm, gt)
    seg = np.arange(t0, t1)
    hit = alarm[seg].sum() > 0
    latency = int(np.argmax(alarm[seg] > 0)) if hit else None

    print("=" * 70)
    print(f"多人聚集仿真  (n={n}, dt={dt}, 半径={GATHER_RADIUS}m, "
          f"时间窗={GATHER_DT}s, >= {GATHER_MIN_PEOPLE} 人)")
    print("=" * 70)
    print(f"聚集段: [{t0},{t1}) = [{t0*dt:.1f}s, {t1*dt:.1f}s)  "
          f"时长 {(t1-t0)*dt:.1f}s")
    print(f"Precision = {p:.3f}   Recall = {r:.3f}   F1 = {f1:.3f}")
    print(f"事件命中 = {hit}   首报延迟 = "
          f"{f'{latency*dt:.2f}s' if latency is not None else '-'}")
    print(f"告警步数 = {int(alarm.sum())}  真值步数 = {int(gt.sum())}")

    # ---- 保存指标 ----
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{
        "dim": "gather(rule)", "precision": round(p, 4),
        "recall": round(r, 4), "f1": round(f1, 4),
        "event_hit": hit, "latency_s": None if latency is None else round(latency * dt, 2),
    }]).to_csv("results/metrics.csv", index=False)

    # ---- 绘图（全英文 · 三合一：≥3 人 / 50 m / 180 s 三条件一图）----
    C = {"main": "#0B6BFF", "near": "#E8272C", "far": "#9AA0A6",
         "zone": "#B2182B", "alarm": "#1B7837"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [1.15, 0.9, 1.0]})

    # (a) 空间规则：局部快照 —— 主行人 + 50 m 圈内的他人 + 圈外的路人
    ax = axes[0]
    ax.add_patch(plt.Circle((0, 0), GATHER_RADIUS, fill=False, ls="--",
                            lw=1.2, color=C["zone"]))
    ax.plot([0], [0], marker="o", ms=9, color=C["main"], label="Main person")
    for i, (dx, dy) in enumerate([(12.0, 0.0), (0.0, 12.0)]):
        ax.plot([dx], [dy], marker="o", ms=9, color=C["near"],
                label=("Other person (within 50 m)" if i == 0 else None))
    ax.plot([60], [60], marker="o", ms=9, color=C["far"], label="Bystander (> 50 m)")
    ax.annotate(f"{GATHER_RADIUS} m radius",
                xy=(GATHER_RADIUS * 0.7071, GATHER_RADIUS * 0.7071),
                xytext=(GATHER_RADIUS * 0.7071 + 16, GATHER_RADIUS * 0.7071 + 16),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.annotate(f"≥ {GATHER_MIN_PEOPLE} people", xy=(6, 6), xytext=(-46, -46),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_xlim(-80, 110); ax.set_ylim(-80, 110)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Spatial rule — ≥ 3 people within 50 m radius", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5); ax.grid(alpha=0.25)

    # (b) 距离随时间 + 50 m 阈值
    ax = axes[1]
    for i, (ox, oy) in enumerate(others):
        d = np.hypot(main_x - ox, main_y - oy)
        ax.plot(t, d, lw=0.8, color=(C["near"] if i < 2 else C["far"]))
    ax.axhline(GATHER_RADIUS, color=C["zone"], ls="--", lw=0.9,
               label=f"{GATHER_RADIUS} m radius")
    ax.axvspan(t0 * dt, t1 * dt, color=C["zone"], alpha=0.12, label="Gathering period")
    ax.set_ylabel("distance to main (m)")
    ax.set_title("Proximity over time", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5)

    # (c) 计数 + 分数 + 180 s 窗
    ax = axes[2]
    ax.step(t, n_near, where="post", lw=1.1, color="0.25",
            label="Co-occurring others ($n_{near}$)")
    ax.plot(t, score, lw=1.2, color=C["main"],
            label=f"score = clip($n_{{near}}$ / {GATHER_MIN_PEOPLE - 1})")
    ax.axhline(GATHER_MIN_PEOPLE - 1, color=C["zone"], ls="--", lw=0.9,
               label=f"≥ {GATHER_MIN_PEOPLE} people  ($n_{{near}}$ = {GATHER_MIN_PEOPLE - 1})")
    ax.axvspan(t0 * dt, t1 * dt, color=C["zone"], alpha=0.12, label="Gathering period")
    ax.fill_between(t, 0, 1, where=alarm > 0, color=C["alarm"], alpha=0.30,
                    step="post", label=f"Alarm (≥ {GATHER_MIN_PEOPLE} people)")
    ax.axvspan(t1 * dt, t1 * dt + GATHER_DT, color="0.5", alpha=0.12)
    ax.text(t1 * dt + GATHER_DT / 2, 0.10, f"{GATHER_DT} s window",
            ha="center", fontsize=8, color="0.3")
    ax.set_ylim(-0.08, max(1.05, n_near.max() + 0.4))
    ax.set_ylabel("count / score")
    ax.set_xlabel("time (s)")
    ax.set_title("Temporal rule — ≥ 3 people within a 180 s window", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5)

    fig.suptitle(f"Multi-person gathering rule  (≥ {GATHER_MIN_PEOPLE} people / "
                 f"{GATHER_RADIUS} m / {GATHER_DT} s)", y=0.998, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_png = "results/simulation.png"
    fig.savefig(out_png, dpi=130)
    print("-" * 70)
    print(f"图片已保存: {os.path.abspath(out_png)}")
    print(f"指标已保存: {os.path.abspath('results/metrics.csv')}")


if __name__ == "__main__":
    main()
