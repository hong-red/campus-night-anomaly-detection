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
"""生成 SDD 8 场景「异常检测」总览图: 横向 1×8, 演示轨迹按最终异常分连续上色
(蓝=正常 → 灰=可疑 → 红=异常), 与 sdd_overview.png 同款版式, 英文标签。

用法: python make_anomaly_figure.py
输出: results/sdd_anomaly_overview.png
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sdd import parse_raw, to_tracks, estimate_scale, pick_demo_sdd, run_track, FPS

# 发散色: 蓝(正常) -> 灰(可疑) -> 红(异常)
CMAP = LinearSegmentedColormap.from_list("anomaly", ["#2a78d6", "#f0efec", "#e34948"])


def main():
    root = "StanfordDroneDataset"
    scenes = sorted([d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and d != ".git"])

    fig, axes = plt.subplots(1, len(scenes), figsize=(len(scenes) * 2.6, 3.7))

    for i, scene in enumerate(scenes):
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
        final = res["final"]

        ax = axes[i]
        ref_jpg = os.path.join(vdir, "reference.jpg")
        if os.path.exists(ref_jpg):
            img = plt.imread(ref_jpg)
            ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.35)

        # 轨迹按最终异常分连续着色
        pts = np.column_stack([dx, dy])
        segs = np.array([pts[:-1], pts[1:]]).transpose(1, 0, 2)
        seg_score = (final[:-1] + final[1:]) / 2.0
        lc = LineCollection(segs, cmap=CMAP, norm=plt.Normalize(0, 1), linewidth=1.8)
        lc.set_array(seg_score)
        ax.add_collection(lc)

        if os.path.exists(ref_jpg):
            ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
        else:
            ax.autoscale()
        ax.set_title(scene, fontsize=11, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # 手动布局: 顶部留标题、底部留 colorbar, 避免重叠遮挡
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.16, wspace=0.02)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.28, 0.05, 0.44, 0.028])   # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Anomaly score  (0 = normal · 1 = anomalous)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Stanford Drone Dataset — anomaly detection (FluxEV + dwell)",
                 fontsize=14, y=0.96)
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sdd_anomaly_overview.png", dpi=200, bbox_inches="tight")
    print("saved results/sdd_anomaly_overview.png")


if __name__ == "__main__":
    main()
