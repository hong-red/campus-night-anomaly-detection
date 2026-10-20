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
Stanford Drone Dataset(SDD) 轨迹 -> FluxEV(短轨迹自适应) + 行为规则, 逐场景看异常检测效果。

数据: StanfordDroneDataset/<scene>/<video>/annotations.txt (像素坐标, 30fps) +
      reference.jpg(该视频第一帧, 作场景"地图"背景)。

坐标: 像素 -> 米。用"行人 bbox 长边(主轴)中位数 ≈ 1.7m 人身高"估计每场景 m/px 比例,
      换算成米制, 使 speed 单位=m/s、与 ETH/UCY、Geolife 管线一致(近似, 误差约 ±30%)。

复用 F:\\仿真跑\\geolife_pipeline 的 FluxEV 核心(SPOT), 仅放宽短轨迹的 GPD 拟合点数下限。
"""
import os, sys, glob
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _fp in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"):
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

REPO = _PKGP.CODE_TSINGHUA
sys.path.insert(0, REPO)
import geolife_pipeline as gp
gp.MIN_FIT_POINTS = 15

V_STAT = 0.4
DWELL_LO, DWELL_HI = 120.0, 300.0
FPS = 30.0
PERSON_LEN_M = 1.7   # 行人(身高)≈1.7m, 用于 px->m 尺度估计

# 类别 -> 大胆配色(取自 dataviz 参考色板, 固定顺序不循环)
CLASS_COLORS = {
    "Pedestrian": "#2a78d6",  # blue
    "Biker":      "#eb6834",  # orange
    "Skater":     "#1baf7a",  # aqua
    "Car":        "#eda100",  # yellow
    "Bus":        "#e87ba4",  # magenta
    "Cart":       "#008300",  # green
}


def parse_raw(path):
    """annotations.txt -> rows: list of dict(tid, xc, yc, w, h, frame, label)

    列: id xmin ymin xmax ymax frame lost occluded generated "label"
    """
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
            label = p[9].strip('"').strip("'")
            lost = int(p[6])
            rows.append(dict(tid=tid, xc=(x1 + x2) / 2.0, yc=(y1 + y2) / 2.0,
                             w=abs(x2 - x1), h=abs(y2 - y1), frame=fr, label=label,
                             lost=lost))
    return rows


def to_tracks(rows, min_pts=15):
    """按 tid 分组, 丢视野外点(lost=1), 取最长连续段 -> list[(x[], y[], t[], label)]。"""
    by = {}
    for r in rows:
        if r.get("lost") == 1:      # 目标在视野外, 位置不可靠
            continue
        a = by.setdefault(r["tid"], ([], [], [], r["label"]))
        a[0].append(r["frame"]); a[1].append(r["xc"]); a[2].append(r["yc"])
    out = []
    for tid, (fr, x, y, label) in by.items():
        order = np.argsort(fr)
        fr = np.asarray(fr)[order]; x = np.asarray(x)[order]; y = np.asarray(y)[order]
        d = np.diff(fr)
        breaks = np.where(d > 2)[0]     # 帧差>2 视为断裂(遮挡/进出视野)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [len(fr)]))
        k = int(np.argmax(ends - starts))
        x = x[starts[k]:ends[k]]; y = y[starts[k]:ends[k]]
        fr = fr[starts[k]:ends[k]]
        if len(x) < min_pts:
            continue
        t = (fr - fr[0]) / FPS
        out.append((x, y, t, label))
    return out


def estimate_scale(rows):
    """行人 bbox 长边中位数 -> m/px。"""
    majors = [max(r["w"], r["h"]) for r in rows if r["label"] == "Pedestrian"]
    if not majors:
        return 0.05   # 兜底
    med = float(np.median(majors))
    return PERSON_LEN_M / med if med > 0 else 0.05


def features_2d(x, y, dt):
    n = len(x)
    d = np.array([np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(n - 1)])
    d = np.concatenate(([0.0], d))
    speed = d / dt
    accel = np.diff(speed, prepend=speed[0]) / dt
    heading = np.zeros(n)
    for i in range(1, n - 1):
        heading[i] = np.arctan2(y[i + 1] - y[i - 1], x[i + 1] - x[i - 1])
    heading[-1] = heading[-2]
    turn = np.concatenate(([0.0], np.abs(np.diff(np.unwrap(heading))))) / dt
    turn[speed < 0.05] = 0.0
    return speed, accel, turn


def dwell_score(speed, dt, v_stat=V_STAT, lo=DWELL_LO, hi=DWELL_HI):
    n = len(speed)
    score = np.zeros(n)
    run = 0.0
    for i in range(n):
        run = run + dt if speed[i] < v_stat else 0.0
        if run <= lo:
            score[i] = 0.0
        elif run >= hi:
            score[i] = 1.0
        else:
            score[i] = (run - lo) / (hi - lo)
    return score


def run_track(x, y, t, dt):
    n = len(x)
    speed, accel, turn = features_2d(x, y, dt)
    train_len = n // 2
    period = max(2, min(int(30.0 / dt), max(2, n // 8)))
    s_w = max(2, min(10, max(2, n // 30)))
    p_w, half_d_w = 3, 1
    ss_idx = s_w * 2 + half_d_w + period * (p_w - 1)
    motion = np.zeros(n)
    motion_ok = 0
    if train_len - ss_idx >= 15:
        for arr in (speed, accel, turn):
            sc, _, _, _ = gp.fluxev_score(
                arr, train_len, period, s_w=s_w, p_w=p_w, half_d_w=half_d_w, q=0.005)
            if np.any(sc > 0):
                motion_ok += 1
            motion[train_len:] = np.maximum(motion[train_len:], sc)
    dwell = dwell_score(speed, dt)
    final = np.maximum(motion, dwell)
    return dict(speed=speed, motion=motion, dwell=dwell, final=final, n=n, dt=dt,
                motion_ok=motion_ok)


def pick_demo_sdd(scene_tracks, label="Pedestrian"):
    """选一条"移动中"且较长的目标轨迹做演示。返回 (x, y, t, label, vdir)。"""
    cands = [s for s in scene_tracks if s[3] == label and len(s[0]) >= 200]
    if not cands:
        cands = [s for s in scene_tracks if len(s[0]) >= 200] or scene_tracks

    def med_speed(s):
        return float(np.median(np.hypot(np.diff(s[0]), np.diff(s[1]))))  # px/frame

    moving = [s for s in cands if 0.5 <= med_speed(s) <= 20.0]   # 明显在动
    if moving:
        return max(moving, key=lambda s: len(s[0]))              # 移动轨迹里选最长
    return max(cands, key=med_speed)                             # 兜底: 取最快


def main():
    os.makedirs("results", exist_ok=True)
    root = _PKGP.SDD_DATA
    scenes = sorted([d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and d != ".git"])
    for scene in scenes:
        videos = sorted(glob.glob(os.path.join(root, scene, "video*", "annotations.txt")))
        if not videos:
            continue
        # 汇总该场景轨迹 + 估算 m/px(跨视频取行人 bbox 主轴中位数)
        all_rows = []
        scene_tracks = []   # (x, y, t, label, vdir)
        for vp in videos:
            rows = parse_raw(vp)
            all_rows.extend(rows)
            vdir = os.path.dirname(vp)
            for x, y, t, label in to_tracks(rows):
                scene_tracks.append((x, y, t, label, vdir))

        if not scene_tracks:
            continue
        scale = estimate_scale(all_rows)
        cnt = Counter(l for _, _, _, l, _ in scene_tracks)
        print(f"\n===== {scene} =====")
        print(f"  视频数={len(videos)}, 轨迹数={len(scene_tracks)}, 类别={dict(cnt)}, "
              f"尺度~{scale:.4f} m/px")

        # 选一条"移动中"的目标做演示
        demo = pick_demo_sdd(scene_tracks)
        dx, dy, dt_, label, vdir = demo
        mx, my = dx * scale, dy * scale

        res = run_track(mx, my, dt_, 1.0 / FPS)
        med = float(np.median(res["speed"]))
        print(f"  演示({label}, {os.path.basename(vdir)}): {res['n']} 点({dt_[-1]/60:.1f} min), "
              f"中位速度={med:.2f} m/s, motion有效={res['motion_ok']}/3, "
              f"异常(≥0.5)点数={int((res['final'] >= 0.5).sum())}")

        # 图1: reference.jpg 背景(压暗) + 全部轨迹按类别大胆配色, 演示轨迹白边加粗高亮
        ref_jpg = os.path.join(vdir, "reference.jpg")
        fig, ax = plt.subplots(figsize=(10, 8))
        if os.path.exists(ref_jpg):
            img = plt.imread(ref_jpg)
            ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.45)
        plotted = set()
        for x, y, t, lb, vd in scene_tracks:
            if vd != vdir:
                continue
            ax.plot(x, y, lw=1.1, alpha=0.9, color=CLASS_COLORS.get(lb, "#e34948"),
                    solid_capstyle="round",
                    label=None if lb in plotted else lb)
            plotted.add(lb)
        ax.plot(dx, dy, lw=4.0, color="white", alpha=0.9, zorder=9, solid_capstyle="round")
        ax.plot(dx, dy, lw=2.2, color=CLASS_COLORS.get(label, "#e34948"), zorder=10,
                solid_capstyle="round", label="Demo track")
        ax.set_title(f"{scene} — trajectories colored by agent class (bg=reference.jpg)",
                     fontsize=13)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(f"results/{scene}_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        # 图2: 异常曲线(米制), 参考色板配色
        fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(dt_, res["speed"], lw=1.4, color="#2a78d6"); axes[0].set_ylabel("speed (m/s)")
        axes[1].plot(dt_, res["motion"], color="#2a78d6", lw=1.4); axes[1].set_ylabel("motion score")
        axes[2].plot(dt_, res["dwell"], color="#eb6834", lw=1.4); axes[2].set_ylabel("dwell score")
        axes[3].plot(dt_, res["final"], color="#e34948", lw=1.8)
        axes[3].axhline(0.5, color="#0b0b0b", ls="--", lw=1.0, label="threshold 0.5")
        axes[3].set_ylabel("final score"); axes[3].set_xlabel("time (s)")
        axes[3].legend(fontsize=8)
        for a in axes:
            a.grid(alpha=0.25, color="#c3c2b7")
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"results/{scene}_anomaly.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    print("\n完成, 输出在 results/ 目录")


if __name__ == "__main__":
    main()
