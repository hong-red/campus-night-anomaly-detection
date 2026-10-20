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
Stanford Drone Dataset 上跑两条行为规则(对应 仿真跑 的「禁区闯入」「多人聚集」)。

  禁区闯入 geofence: 在场景里定义一块虚拟禁区(圆), 找一条真实行人轨迹穿越它 -> 0→1→0。
  多人聚集 gather:   真实场景里, 每个采样时刻统计「R 米内互相靠近的 ≥3 人」最大簇 -> 检测聚集。

输出:
  results/sdd_geofence.png
  results/sdd_gather.png
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sdd import parse_raw, to_tracks, estimate_scale, FPS

ZONE_RADIUS_M = 12.0       # 禁区半径(米)
GATHER_RADIUS_M = 5.0      # 聚集半径(米)
GATHER_MIN_PEOPLE = 3      # 深夜 >=3 人


def pick_video(scene):
    """选该场景轨迹数最多的视频, 返回 (vdir, rows, tracks, scale, img)。"""
    root = _PKGP.SDD_DATA
    videos = sorted(glob.glob(os.path.join(root, scene, "video*", "annotations.txt")))
    best_vp, best_rows = None, None
    for vp in videos:
        rows = parse_raw(vp)
        if best_vp is None or len(to_tracks(rows)) > len(to_tracks(best_rows)):
            best_vp, best_rows = vp, rows
    vdir = os.path.dirname(best_vp)
    scale = estimate_scale(best_rows)
    ref = os.path.join(vdir, "reference.jpg")
    img = plt.imread(ref) if os.path.exists(ref) else None
    return vdir, best_rows, to_tracks(best_rows), scale, img


# --------------------------- 禁区闯入 ---------------------------

def run_geofence(scene="deathCircle"):
    vdir, rows, trs, scale, img = pick_video(scene)
    if img is None:
        print("[geofence] 无参考图"); return
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    r_px = ZONE_RADIUS_M / scale

    # 找一条「穿入又穿出」(起终点在圈外、中间进圈)且够长的行人轨迹
    best = None
    for x, y, t, label in trs:
        if label != "Pedestrian" or len(x) < 60:
            continue
        inside = (x - cx) ** 2 + (y - cy) ** 2 <= r_px ** 2
        frac = float(inside.mean())
        if not (0.15 <= frac <= 0.85):
            continue
        if inside[0] or inside[-1]:      # 起终点必须在圈外, 才是"闯入后离开"
            continue
        if best is None or len(x) > len(best[0]):
            best = (x, y, t, inside)
    if best is None:
        print("[geofence] 没找到穿越禁区的轨迹"); return
    x, y, t, inside = best

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.imshow(img, extent=[0, W, H, 0], alpha=0.35)
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot(cx + r_px * np.cos(th), cy + r_px * np.sin(th), ls="--", lw=1.8,
            color="#e34948", label="Restricted zone")
    ax.fill(cx + r_px * np.cos(th), cy + r_px * np.sin(th), color="#e34948", alpha=0.08)
    ax.plot(x[~inside], y[~inside], lw=1.0, color="#898781", label="Trajectory (outside)")
    ax.plot(x[inside], y[inside], lw=2.2, color="#e34948", label="Trajectory (inside = breach)")
    ax.scatter([x[0]], [y[0]], s=45, color="#1baf7a", zorder=5, label="Start")
    ax.scatter([x[-1]], [y[-1]], s=45, color="#2a78d6", zorder=5, label="End")
    ax.set_title(f"{scene} — restricted-zone breach (real SDD trajectory)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_axis_off()

    ax = axes[1]
    alarm = inside.astype(int)
    ax.step(t, alarm, where="post", lw=1.8, color="#e34948", label="Alarm")
    ax.fill_between(t, 0, 1, where=alarm > 0, step="post", color="#e34948", alpha=0.3)
    ax.set_ylim(-0.05, 1.15); ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Alarm"])
    ax.set_xlabel("time (s)")
    ax.set_title("Time-warning (alarm over time)", fontsize=12)
    ax.grid(alpha=0.2, color="#c3c2b7")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sdd_geofence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_in = int(inside.sum())
    n_cross = int(np.sum(np.diff(inside.astype(int)) != 0)) // 2
    print(f"[geofence] {scene}: 轨迹 {len(x)} 点({t[-1]:.0f}s), 禁区内 {n_in} 点"
          f"({n_in / len(x) * 100:.0f}%), 穿入穿出 {n_cross} 次")


# --------------------------- 多人聚集 ---------------------------

def run_gather(scene="nexus", step=15):
    vdir, rows, trs, scale, img = pick_video(scene)
    if img is None:
        print("[gather] 无参考图"); return
    H, W = img.shape[:2]
    R = GATHER_RADIUS_M / scale

    # 按帧分组行人位置
    by_frame = {}
    for r in rows:
        if r["label"] == "Pedestrian":
            by_frame.setdefault(r["frame"], []).append((r["xc"], r["yc"]))
    frames = sorted(by_frame.keys())
    frames = frames[::step]          # 每 step 帧采样一次

    times, max_near = [], []
    peak = None
    for f in frames:
        pos = np.array(by_frame[f])
        t = (f - frames[0]) / FPS
        times.append(t)
        if len(pos) < 2:
            max_near.append(0); continue
        mx = 0
        for i in range(len(pos)):
            d = np.hypot(pos[:, 0] - pos[i, 0], pos[:, 1] - pos[i, 1])
            mx = max(mx, int((d <= R).sum()) - 1)   # 除自己外靠近的他人数
        max_near.append(mx)
        if mx >= GATHER_MIN_PEOPLE - 1 and peak is None:
            peak = (f, pos, mx)
    times = np.array(times)
    max_near = np.array(max_near)
    score = np.clip(max_near / (GATHER_MIN_PEOPLE - 1), 0.0, 1.0)
    alarm = (max_near >= GATHER_MIN_PEOPLE - 1).astype(int)

    n_alarm = int(alarm.sum())
    peak_mx = int(max_near.max()) if len(max_near) else 0
    print(f"[gather] {scene}: 采样帧 {len(frames)}, 峰值聚集 "
          f"{peak_mx + 1} 人, 聚集帧 {n_alarm} ({n_alarm / max(len(frames), 1) * 100:.0f}%)")

    # 图: 左=峰值聚集空间快照, 右=时间预警图(报警时间线)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.imshow(img, extent=[0, W, H, 0], alpha=0.35)
    if peak is not None:
        f, pos, mx = peak
        pts = np.array(pos)
        center_i = 0
        for i in range(len(pts)):
            d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            if int((d <= R).sum()) - 1 == mx:
                center_i = i; break
        cx0, cy0 = pts[center_i]
        circ = plt.Circle((cx0, cy0), R, fill=False, ls="--", lw=1.8, color="#e34948")
        ax.add_patch(circ)
        near = np.hypot(pts[:, 0] - cx0, pts[:, 1] - cy0) <= R
        ax.scatter(pts[~near, 0], pts[~near, 1], s=22, color="#898781",
                   label="Others (> 5 m)")
        ax.scatter(pts[near, 0], pts[near, 1], s=34, color="#e34948",
                   label=f"Gathering (≥ {GATHER_MIN_PEOPLE} people)")
        ax.scatter([cx0], [cy0], s=55, color="#2a78d6", zorder=5, label="Reference person")
    ax.set_title(f"{scene} — snapshot at peak gathering ({peak_mx + 1} people)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_axis_off()

    ax = axes[1]
    ax.step(times, alarm, where="post", lw=1.8, color="#e34948", label="Alarm")
    ax.fill_between(times, 0, 1, where=alarm > 0, step="post", color="#e34948", alpha=0.3)
    ax.set_ylim(-0.05, 1.15); ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Alarm"])
    ax.set_xlabel("time (s)")
    ax.set_title("Time-warning (alarm over time)", fontsize=12)
    ax.grid(alpha=0.2, color="#c3c2b7")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sdd_gather.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_geofence()
    run_gather()
