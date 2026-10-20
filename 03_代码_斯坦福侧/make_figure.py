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
"""生成 Stanford Drone Dataset 8 场景总览图: 横向 1×8 多子图, 英文标签, 轨迹按类别大胆配色。

用法: python make_figure.py
输出: results/sdd_overview.png
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 类别 -> 大胆配色(dataviz 参考色板, 固定顺序不循环)
CLASS_COLORS = {
    "Pedestrian": "#2a78d6",  # blue
    "Biker":      "#eb6834",  # orange
    "Skater":     "#1baf7a",  # aqua
    "Car":        "#eda100",  # yellow
    "Bus":        "#e87ba4",  # magenta
    "Cart":       "#008300",  # green
}
CLASS_ORDER = ["Pedestrian", "Biker", "Skater", "Car", "Bus", "Cart"]


def parse_raw(path):
    """annotations.txt -> rows(dict: tid,x,y,frame,label), 丢 lost(视野外)点。"""
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 10:
                continue
            try:
                tid = int(p[0])
                x1, y1, x2, y2 = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                fr = int(p[5])
            except ValueError:
                continue
            if int(p[6]) == 1:      # lost=1, 位置不可靠
                continue
            label = p[9].strip('"').strip("'")
            rows.append(dict(tid=tid, x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0,
                             frame=fr, label=label))
    return rows


def to_tracks(rows, min_pts=8):
    """按 tid 分组, 取最长连续段 -> list[(x[], y[], label)]。"""
    by = {}
    for r in rows:
        a = by.setdefault(r["tid"], ([], [], [], r["label"]))
        a[0].append(r["frame"]); a[1].append(r["x"]); a[2].append(r["y"])
    out = []
    for tid, (fr, x, y, label) in by.items():
        order = np.argsort(fr)
        fr = np.asarray(fr)[order]; x = np.asarray(x)[order]; y = np.asarray(y)[order]
        d = np.diff(fr)
        breaks = np.where(d > 2)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [len(fr)]))
        k = int(np.argmax(ends - starts))
        x = x[starts[k]:ends[k]]; y = y[starts[k]:ends[k]]
        if len(x) < min_pts:
            continue
        out.append((x, y, label))
    return out


def main():
    root = "StanfordDroneDataset"
    scenes = sorted([d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and d != ".git"])

    fig, axes = plt.subplots(1, len(scenes), figsize=(len(scenes) * 2.4, 3.6))
    all_labels = set()

    for i, scene in enumerate(scenes):
        videos = sorted(glob.glob(os.path.join(root, scene, "video*", "annotations.txt")))
        # 选该场景轨迹数最多的视频作代表
        best_vp, best_n = None, -1
        for vp in videos:
            trs = to_tracks(parse_raw(vp))
            if len(trs) > best_n:
                best_n, best_vp = len(trs), vp
        if best_vp is None:
            continue
        trs = to_tracks(parse_raw(best_vp))
        vdir = os.path.dirname(best_vp)
        ref_jpg = os.path.join(vdir, "reference.jpg")

        ax = axes[i]
        if os.path.exists(ref_jpg):
            img = plt.imread(ref_jpg)
            ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.35)
        for x, y, label in trs:
            all_labels.add(label)
            ax.plot(x, y, lw=0.8, alpha=0.85, color=CLASS_COLORS.get(label, "#e34948"),
                    solid_capstyle="round")
        ax.set_title(scene, fontsize=11, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # 图例(按固定顺序)
    ordered = [l for l in CLASS_ORDER if l in all_labels]
    handles = [plt.Line2D([0], [0], color=CLASS_COLORS[l], lw=2.2) for l in ordered]
    fig.legend(handles, ordered, loc="lower center", ncol=len(ordered),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03),
               columnspacing=1.0, handlelength=1.2)
    fig.suptitle("Stanford Drone Dataset — trajectories by agent class",
                 fontsize=14, y=1.04)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sdd_overview.png", dpi=200, bbox_inches="tight")
    print("saved results/sdd_overview.png")
    print("scenes:", scenes)
    print("classes:", ordered)


if __name__ == "__main__":
    main()
