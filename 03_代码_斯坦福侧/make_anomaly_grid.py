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
"""把 8 个场景的 anomaly 结果拼成一张 3×8 多子图大图:
   speed(上下文) / motion score(异常分) / alarm(二值告警)。

   注: dwell(>5min 逗留) 在此省略 —— SDD 片段仅 1~6min 且演示轨迹为行走行人,
   dwell 恒为 0; 故 final = max(motion, dwell) = motion, 也一并省略。
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sdd import parse_raw, to_tracks, estimate_scale, pick_demo_sdd, run_track, FPS


def main():
    root = "StanfordDroneDataset"
    scenes = sorted([d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and d != ".git"])
    fig, axes = plt.subplots(3, len(scenes), figsize=(len(scenes) * 2.3, 3 * 2.2),
                             sharex="col")

    for j, scene in enumerate(scenes):
        videos = sorted(glob.glob(os.path.join(root, scene, "video*", "annotations.txt")))
        all_rows, scene_tracks = [], []
        for vp in videos:
            rows = parse_raw(vp)
            all_rows.extend(rows)
            vdir = os.path.dirname(vp)
            for x, y, t, label in to_tracks(rows):
                scene_tracks.append((x, y, t, label, vdir))
        if not scene_tracks:
            continue
        scale = estimate_scale(all_rows)
        demo = pick_demo_sdd(scene_tracks)
        dx, dy, dt_, label, vdir = demo
        res = run_track(dx * scale, dy * scale, dt_, 1.0 / FPS)
        t = dt_
        speed = res["speed"]
        motion = res["motion"]
        alarm = (res["final"] >= 0.5).astype(int)

        # 行0: speed(上下文, 灰色)
        ax = axes[0, j]
        ax.plot(t, speed, lw=1.0, color="#898781")
        # 行1: motion score(蓝色) + 0.5 阈值
        ax = axes[1, j]
        ax.plot(t, motion, lw=1.2, color="#2a78d6")
        ax.axhline(0.5, color="#0b0b0b", ls="--", lw=0.9)
        ax.set_ylim(0, 1)
        # 行2: alarm(红色阶梯)
        ax = axes[2, j]
        ax.plot(t, alarm, lw=1.0, color="#e34948", drawstyle="steps-post")
        ax.set_ylim(-0.05, 1.3)
        ax.set_yticks([0, 1])

        axes[0, j].set_title(scene, fontsize=11, pad=4)
        for i in range(3):
            axes[i, j].tick_params(labelsize=7)

    # 行 y 标签(最左列)
    axes[0, 0].set_ylabel("speed (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("motion score", fontsize=9)
    axes[2, 0].set_ylabel("alarm", fontsize=9)
    # 列 x 标签(最底行)
    for j in range(len(scenes)):
        axes[2, j].set_xlabel("time (s)", fontsize=9)

    fig.suptitle("Stanford Drone Dataset — per-scene anomaly detection (FluxEV motion)",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sdd_anomaly_grid.png", dpi=200, bbox_inches="tight")
    print("saved results/sdd_anomaly_grid.png")


if __name__ == "__main__":
    main()
