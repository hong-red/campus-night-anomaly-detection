#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强约束 if 规则(禁区闯入 / 多人聚集) 的准确率 + 相对运动模型的优势实验
=====================================================================

实验设计：
  1) 禁区闯入 —— if(point_in_polygon)：M 条匀速穿越 BIGC 校区的合成轨迹，
     真值=多边形内外(几何定义)，规则判定 vs FluxEV 运动分数做对照。
     论据：穿越围栏不改变运动模式，运动模型对"内外"无判别力。
  2) 多人聚集 —— if(>=3人 / 50m / 180s)：主行人匀速直行，2 人贴靠+1 路人，
     真值=聚集时段，规则判定 vs FluxEV 运动分数做对照。
     论据：主行人全程匀速(运动分数恒 0)，聚集是跨轨迹空间共现，运动模型看不到。

输出：
  results/rule_advantage_geofence.png   禁区闯入（地图 + 时序对照）
  results/rule_advantage_gather.png     多人聚集（复用 simulate 场景）
  results/rule_advantage_summary.png    规则 vs 运动模型 AUROC 汇总
  results/rule_advantage.json           全量指标
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics

from geolife_pipeline import (load_geofence_polygon, geofence_score_polygon,
                              run_pipeline)
from simulate import build_scene, gather_score, GATHER_RADIUS, GATHER_DT, GATHER_MIN_PEOPLE

os.makedirs("results", exist_ok=True)

# 项目配色
C_RULE = "#1B7837"      # 规则（if）
C_MOTION = "#9AA0A6"    # 运动模型
C_IN = "#E8272C"        # 越界/聚集真值
C_GT = "#FF7A00"


def auroc(gt, score):
    """算 AUROC；分数无方差/类别不齐时返回 0.5(无判别力)。"""
    if (gt == 1).any() and (gt == 0).any() and np.std(score) > 1e-9:
        return float(metrics.roc_auc_score(gt, score))
    return 0.5


# ============================ 1) 禁区闯入 ============================
poly = load_geofence_polygon("geofence_bigc.json")
lats = [p[0] for p in poly]
lons = [p[1] for p in poly]
lat0 = (min(lats) + max(lats)) / 2
lon0 = (min(lons) + max(lons)) / 2
kx = 111320.0 * np.cos(np.radians(lat0))
ky = 110540.0
rng = np.random.default_rng(2)

M = 40
dt = 5.0
speed = 1.4
rule_all, gt_all, mot_all = [], [], []
demo = None
for k in range(M):
    lat_c = rng.uniform(min(lats), max(lats))
    x0 = (min(lons) - lon0) * kx - 300.0
    x1 = (max(lons) - lon0) * kx + 300.0
    yc = (lat_c - lat0) * ky
    L = x1 - x0
    n = max(int(L / speed / dt), 40)
    xs = np.linspace(x0, x1, n)
    ys = np.full(n, yc)
    lon = lon0 + xs / kx
    lat = lat0 + ys / ky
    t = np.arange(n) * dt
    rule = geofence_score_polygon(lat, lon, poly)
    res = run_pipeline(lat, lon, t, dt, poly=None)   # 不喂围栏，只取运动分数
    rule_all.append(rule)
    gt_all.append(rule.copy())                        # 几何真值 = 多边形内外
    mot_all.append(res["motion"])
    if demo is None and rule.sum() > 0 and (1 - rule).sum() > 0:
        demo = (lon, lat, t, rule, res["motion"])

rule_all = np.concatenate(rule_all)
gt_all = np.concatenate(gt_all)
mot_all = np.concatenate(mot_all)
geo_acc = float((rule_all == gt_all).mean())
geo_rule_auc = auroc(gt_all, rule_all)
geo_motion_auc = auroc(gt_all, mot_all)

# ============================ 2) 多人聚集 ============================
n = 2400
dtg = 0.5
mx, my, others, tg, t0, t1 = build_scene(n, dtg, rng)
score, nn = gather_score(mx, my, others, dtg)
alarm = (nn >= GATHER_MIN_PEOPLE - 1).astype(int)
gt = np.zeros(n, int)
gt[t0:t1] = 1

tp = int(((alarm == 1) & (gt == 1)).sum())
fp = int(((alarm == 1) & (gt == 0)).sum())
fn = int(((alarm == 0) & (gt == 1)).sum())
tn = int(((alarm == 0) & (gt == 0)).sum())
g_prec = tp / (tp + fp)
g_rec = tp / (tp + fn)
g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec)
g_acc = (tp + tn) / n
g_rule_auc = auroc(gt, score)
g_rule_score_auc = auroc(gt, score)

# 主行人 -> lat/lon（匀速直行），跑 FluxEV 运动分数，看它能否"看见"聚集
la = 40.0 + my / 110540.0
lo = 116.3 + mx / (111320.0 * np.cos(np.radians(40.0)))
tg_arr = np.arange(n) * dtg
resg = run_pipeline(la, lo, tg_arr, dtg, poly=None)
motion_g = resg["motion"]
g_motion_auc = auroc(gt, motion_g)

# ============================ 结果汇总 ============================
res_out = {
    "geofence_intrusion": {
        "rule": {"accuracy": round(geo_acc, 4), "auroc": round(geo_rule_auc, 4)},
        "motion_fluxev": {"auroc": round(geo_motion_auc, 4)},
        "n_traj": M,
    },
    "multi_person_gathering": {
        "rule": {"precision": round(g_prec, 4), "recall": round(g_rec, 4),
                 "f1": round(g_f1, 4), "accuracy": round(g_acc, 4),
                 "auroc": round(g_rule_score_auc, 4)},
        "motion_fluxev": {"auroc": round(g_motion_auc, 4)},
    },
}
json.dump(res_out, open("results/rule_advantage.json", "w"), ensure_ascii=False, indent=2)

print("=" * 78)
print("强约束 if 规则 vs 运动模型(FluxEV) 对照实验")
print("=" * 78)
print(f"[禁区闯入]  规则 accuracy = {geo_acc:.4f}  规则 AUROC = {geo_rule_auc:.3f}"
      f"  |  运动模型 AUROC = {geo_motion_auc:.3f}")
print(f"[多人聚集]  规则 P={g_prec:.3f} R={g_rec:.3f} F1={g_f1:.3f} "
      f"acc={g_acc:.3f} AUROC={g_rule_score_auc:.3f}  |  运动模型 AUROC = {g_motion_auc:.3f}")

# ============================ 图1 禁区闯入 ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
px = [p[1] for p in poly] + [poly[0][1]]
py = [p[0] for p in poly] + [poly[0][0]]
ax.plot(px, py, "-", lw=1.6, color=C_IN, label="Restricted polygon")
ax.fill(px, py, color=C_IN, alpha=0.08)
lon_d, lat_d, t_d, rule_d, mot_d = demo
inside = rule_d > 0
ax.plot(lon_d[~inside], lat_d[~inside], lw=1.0, color="0.6", label="Outside")
ax.plot(lon_d[inside], lat_d[inside], lw=1.8, color=C_IN, label="Inside (intrusion)")
ax.scatter(lon_d[0], lat_d[0], s=50, color="#00B140", zorder=5, label="Start")
ax.scatter(lon_d[-1], lat_d[-1], s=50, color="#0B6BFF", zorder=5, label="End")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Geofence: a constant-speed walker crosses the fence", fontsize=9.5, loc="left")
ax.legend(fontsize=7.5, loc="best"); ax.grid(alpha=0.3)

ax = axes[1]
ax.step(t_d, rule_d, where="post", color=C_RULE, lw=1.4, label="if(point-in-polygon)")
ax.plot(t_d, mot_d, color=C_MOTION, lw=1.3, label="FluxEV motion score")
ax.axhline(0.5, color="0.6", ls="--", lw=0.8)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Score")
ax.set_ylim(-0.05, 1.15)
ax.set_title(f"Rule exact (acc={geo_acc:.2f}) vs motion blind (AUROC={geo_motion_auc:.2f})",
             fontsize=9.5, loc="left")
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/rule_advantage_geofence.png", dpi=200)
plt.close(fig)

# ============================ 图2 多人聚集 ============================
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
ax = axes[0]
ax.plot(tg, np.hypot(mx - others[0][0], my - others[0][1]), color=C_IN, lw=0.9, label="gatherer A distance")
ax.plot(tg, np.hypot(mx - others[1][0], my - others[1][1]), color=C_IN, lw=0.9, ls="-.", label="gatherer B distance")
ax.plot(tg, np.hypot(mx - others[2][0], my - others[2][1]), color="0.6", lw=0.9, label="bystander distance")
ax.axhline(GATHER_RADIUS, color=C_GT, ls="--", lw=0.9, label=f"{GATHER_RADIUS} m radius")
ax.axvspan(t0 * dtg, t1 * dtg, color=C_GT, alpha=0.12, label="Gathering period (GT)")
ax.set_ylabel("distance to main (m)")
ax.set_title("Gathering: proximity over time", fontsize=9.5, loc="left")
ax.legend(fontsize=7.5, loc="upper left", ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
ax.step(tg, alarm, where="post", color=C_RULE, lw=1.4, label="if(>=3 people / 50 m / 180 s)")
ax.plot(tg, mot := motion_g, color=C_MOTION, lw=1.2, label="FluxEV motion score (main person)")
ax.axvspan(t0 * dtg, t1 * dtg, color=C_GT, alpha=0.12, label="Gathering period (GT)")
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel("Time (s)"); ax.set_ylabel("score / alarm")
ax.set_title(f"Rule P={g_prec:.2f} R={g_rec:.2f} F1={g_f1:.2f}  vs  motion blind (AUROC={g_motion_auc:.2f})",
             fontsize=9.5, loc="left")
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/rule_advantage_gather.png", dpi=200)
plt.close(fig)

# ============================ 图3 汇总 ============================
fig, ax = plt.subplots(figsize=(6.8, 3.8))
tasks = ["Geofence\nintrusion", "Multi-person\ngathering"]
rule_auc = [geo_rule_auc, g_rule_score_auc]
motion_auc = [geo_motion_auc, g_motion_auc]
x = np.arange(2); w = 0.34
b1 = ax.bar(x - w / 2, rule_auc, w, color=C_RULE, label="if-rule", zorder=3)
b2 = ax.bar(x + w / 2, motion_auc, w, color=C_MOTION, label="FluxEV motion (learned-free EVT)", zorder=3)
for xi, v in zip(x - w / 2, rule_auc):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, motion_auc):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.axhline(0.5, color="0.6", ls="--", lw=0.9)
ax.text(1.17, 0.515, "chance 0.5", ha="right", va="bottom", fontsize=7.5, color="0.4")
ax.set_xticks(x); ax.set_xticklabels(tasks)
ax.set_ylabel("AUROC")
ax.set_ylim(0, 1.15)
ax.set_title("Hard if-conditions detect spatial properties motion models cannot",
             fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("results/rule_advantage_summary.png", dpi=200)
plt.close(fig)

print("-" * 78)
print("已保存: results/rule_advantage_geofence.png / _gather.png / _summary.png / rule_advantage.json")
