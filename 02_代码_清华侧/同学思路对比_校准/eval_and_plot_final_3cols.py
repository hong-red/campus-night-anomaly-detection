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
"""六行为 × 三栏(硬 / 软 / 硬加软) 准确率 —— 本地软(TCN)版本 + 分组图。

与之前的差异（用户要求「其他的不变，只把软换成本地跑」）：
  硬     = if 规则 max(dwell, gather, geofence)，dwell 用提升后的 lo=15s
          （hard_accuracy_6behaviors.py 同款注入/口径）
  软     = 本地 TCN checkpoint（z-clip S_dl≥0.5 acc@0.5），不再用同学 GPU 汇总表
  硬加软 = 逐帧 max(硬分数, 软分数) 后再算 acc@0.5（真实融合，非准确率取 max）

行为 → 负责分支（最终架构，图里分组着色）：
  if-else        Intrusion(禁区闯入) / Gathering(多人聚集)
  if-else+FluxEV Lingering(原地逗留)
  TCN            Wandering(徘徊) / Sprint(疾跑) / Zigzag(之字形折返)

输出：
  final_3cols_local.json
  figs/fig_六行为_硬软硬加软_本地软.png
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = _PKGP.CODE_TSINGHUA
for _p in (_PROJ, os.path.join(_PROJ, "tcn_dl_branch"),
           _PKGP.STYLE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows 必须先 import torch 再 import numpy（c10.dll 冲突）
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics

import geolife_pipeline as gp
import tcn_branch as T
from style import apply_cn_style, save

apply_cn_style()

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0
DWELL_LO = 15            # 提升后的 dwell 起算阈值（秒）
DWELL_SEC = 360          # dwell 注入时长
DWELL_HI = gp.DWELL_MIN  # 300s，满 5 分钟 -> 分数 1

# ---------------- 坐标转换 ----------------

def local_xy(lat, lon):
    lat0, lon0 = lat[0], lon[0]
    x = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110540.0
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / 110540.0
    lon = lon0 + x / (111320.0 * np.cos(np.radians(lat0)))
    return lat, lon


def max_stationary_run(speed, dt):
    run = best = 0.0
    for v in speed:
        if v < gp.V_STATIONARY:
            run += dt
            best = max(best, run)
        else:
            run = 0.0
    return best


# ---------------- 运动注入器（与 tcn_branch 一致） ----------------

def inject_loiter(x, y, i0, i1, amp=8.0, cycle=16):
    cx, cy = float(np.mean(x[i0:i1])), float(np.mean(y[i0:i1]))
    dx, dy = x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]
    norm = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / norm, dy / norm
    idx = np.arange(i0, i1)
    ph = np.sin(2 * np.pi * (idx - i0) / cycle)
    x[i0:i1] = cx + ux * amp * ph
    y[i0:i1] = cy + uy * amp * ph


def inject_run(x, y, i0, i1, factor=2.5):
    dx, dy = x - x[i0], y - y[i0]
    x[i0:i1] = x[i0] + dx[i0:i1] * factor
    y[i0:i1] = y[i0] + dy[i0:i1] * factor


def inject_zigzag(x, y, i0, i1, amp=5.0, cycle=20):
    dx, dy = x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]
    norm = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    idx = np.arange(i0, i1)
    osc = amp * np.sin(2 * np.pi * (idx - i0) / cycle)
    x[i0:i1] += px * osc
    y[i0:i1] += py * osc


# ---------------- 行为注入 ----------------

def inject_dwell(lat, lon, n):
    i0 = int(n * 0.55); i1 = min(i0 + int(DWELL_SEC / DT), n - 2)
    lat = lat.copy(); lon = lon.copy()
    lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
    gt = np.zeros(n); gt[i0:i1] = 1
    return lat, lon, None, None, gt


def inject_geofence(lat, lon, n):
    c = int(n * 0.7)
    zone = (float(lat[c]), float(lon[c]), 60.0)
    gt = (gp.geofence_score(lat, lon, zone) > 0).astype(int)
    return lat, lon, None, zone, gt


def inject_gather(lat, lon, n):
    g0 = int(n * 0.55); g1 = min(g0 + int(300 / DT), n - 1)
    kx = 111320.0 * np.cos(np.radians(lat[g0])); ky = 110540.0
    others = []
    for dx, dy in ((12.0, 0.0), (0.0, 12.0)):
        o_lat = np.full(n, lat[g0] + 200.0 / ky)
        o_lon = np.full(n, lon[g0] + 200.0 / kx)
        o_lat[g0:g1] = lat[g0:g1] + dy / ky
        o_lon[g0:g1] = lon[g0:g1] + dx / kx
        others.append((o_lat, o_lon))
    gt = np.zeros(n); gt[g0:g1] = 1
    return lat, lon, others, None, gt


def inject_motion(lat, lon, n, kind):
    x, y = local_xy(lat, lon)
    i0 = int(n * 0.55); i1 = min(i0 + max(16, int(n * 0.18)), n - 1)
    if kind == "wandering":
        inject_loiter(x, y, i0, i1)
    elif kind == "sprint":
        inject_run(x, y, i0, i1)
    else:
        inject_zigzag(x, y, i0, i1)
    lat2, lon2 = xy_to_latlon(x, y, lat[0], lon[0])
    gt = np.zeros(n); gt[i0:i1] = 1
    return lat2, lon2, None, None, gt


# ---------------- 打分 ----------------

def hard_score(lat, lon, dt, others, zone):
    """硬 = if 规则 max(dwell[15s], gather, geofence)，不含 EVT 运动极值。"""
    speed, _, _ = gp.trajectory_features(lat, lon, dt)
    dwell = gp.stationary_episode_score(speed, dt, DWELL_LO, DWELL_HI)
    gather = gp.gather_score(lat, lon, others, dt) if others else np.zeros(len(lat))
    geof = gp.geofence_score(lat, lon, zone) if zone else np.zeros(len(lat))
    return np.maximum.reduce([dwell, gather, geof])


def frame_ade(model, x, y, obs_len, pred_len):
    """整条轨迹逐帧原始 ADE(米)，无窗口处为 NaN。"""
    n = len(x)
    total = obs_len + pred_len
    if n < total:
        return np.full(n, np.nan)
    obs_list, fut_list = [], []
    for s in range(0, n - total + 1):
        ref = np.array([x[s + obs_len - 1], y[s + obs_len - 1]])
        obs_list.append(np.stack([x[s:s + obs_len], y[s:s + obs_len]], -1) - ref)
        fut_list.append(np.stack([x[s + obs_len:s + total], y[s + obs_len:s + total]], -1) - ref)
    ade_all = T.compute_ade(model, np.asarray(obs_list, np.float32),
                            np.asarray(fut_list, np.float32))
    ade = np.full(n, np.nan)
    ade[obs_len:obs_len + len(ade_all)] = ade_all
    return ade


def zclip(ade, mu, sigma, k=3.0):
    ade = np.nan_to_num(ade, nan=mu)
    return np.clip((ade - mu) / (k * sigma + 1e-8), 0.0, 1.0)


BEHAVIORS = [
    ("Intrusion", "geofence", "if-else"),
    ("Gathering", "gather", "if-else"),
    ("Lingering", "dwell", "if-else+FluxEV"),
    ("Wandering", "wandering", "TCN"),
    ("Sprint", "sprint", "TCN"),
    ("Zigzag", "zigzag", "TCN"),
]

GROUP_CN = {"if-else": "if-else 规则", "if-else+FluxEV": "if-else + FluxEV",
            "TCN": "TCN"}
GROUP_COLOR = {"if-else": "#1B7837", "if-else+FluxEV": "#4393C3", "TCN": "#eb6834"}


def pool_metrics(g_pool, s_pool):
    g = np.concatenate(g_pool)
    s = np.concatenate(s_pool)
    pred = (s >= 0.5).astype(int)
    acc = float((pred == g).mean())
    tp = int(((pred == 1) & (g == 1)).sum())
    fp = int(((pred == 1) & (g == 0)).sum())
    fn = int(((pred == 0) & (g == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auroc = (metrics.roc_auc_score(g, s)
             if g.sum() not in (0, len(g)) and np.std(s) > 1e-9 else None)
    return dict(acc=round(acc, 4), prec=round(prec, 4), rec=round(rec, 4),
                f1=round(f1, 4), auroc=round(auroc, 4) if auroc is not None else None,
                neg_frac=round(float((g == 0).mean()), 4))


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    print(f"TCN: obs={obs} pred={pred} mu={mu:.2f}m sigma={sigma:.2f}m")

    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    clean = []
    for w in walks:
        lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
        speed0, _, _ = gp.trajectory_features(lat, lon, DT)
        if max_stationary_run(speed0, DT) <= 120:
            clean.append(w)
    print(f"过滤自然长停留(>120s)后 = {len(clean)} 条轨迹")

    result = {}
    for label, key, grp in BEHAVIORS:
        pools = {"hard": ([], []), "soft": ([], []), "fused": ([], [])}
        for w in clean:
            lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
            n = len(lat)
            if n < int(700 / DT):
                continue
            if key == "geofence":
                lat, lon, others, zone, gt = inject_geofence(lat, lon, n)
            elif key == "gather":
                lat, lon, others, zone, gt = inject_gather(lat, lon, n)
            elif key == "dwell":
                lat, lon, others, zone, gt = inject_dwell(lat, lon, n)
            else:
                lat, lon, others, zone, gt = inject_motion(lat, lon, n, key)

            hard = hard_score(lat, lon, DT, others, zone)
            x, y = T.latlon_to_local(lat, lon)
            soft = zclip(frame_ade(model, x, y, obs, pred), mu, sigma)
            fused = np.maximum(hard, soft)

            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            for col, s in (("hard", hard), ("soft", soft), ("fused", fused)):
                pools[col][0].append(g)
                pools[col][1].append(s[mask])

        row = {}
        for col in ("hard", "soft", "fused"):
            row[col] = pool_metrics(pools[col][0], pools[col][1])
        result[key] = dict(label=label, group=grp, **row)

        h, s, f = row["hard"], row["soft"], row["fused"]
        print(f"{label:<10} [{grp:<12}] 硬 acc={h['acc']:.3f}(A{h['auroc']})  "
              f"软 acc={s['acc']:.3f}(A{s['auroc']})  硬加软 acc={f['acc']:.3f}(A{f['auroc']})  "
              f"P={f['prec']:.3f} R={f['rec']:.3f} F1={f['f1']:.3f}")

    out_path = os.path.join(_HERE, "final_3cols_local.json")
    json.dump(result, open(out_path, "w"), ensure_ascii=False, indent=2)
    print("\nsaved", out_path)

    # ================= 图 =================
    plot(result)


def plot(result):
    keys = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag"]
    labels = ["Intrusion\n禁区闯入", "Gathering\n多人聚集", "Lingering\n原地逗留",
              "Wandering\n徘徊", "Sprint\n疾跑", "Zigzag\n之字形折返"]
    groups = [result[k]["group"] for k in keys]
    hard_v = [result[k]["hard"]["acc"] for k in keys]
    soft_v = [result[k]["soft"]["acc"] for k in keys]
    fused_v = [result[k]["fused"]["acc"] for k in keys]
    neg_frac = [result[k]["hard"]["neg_frac"] for k in keys]

    BAR = {"hard": "#2C7BB6", "soft": "#EE7733", "fused": "#762A83"}

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(keys))
    w = 0.24

    ax.bar(x - w, hard_v, w, color=BAR["hard"], label="硬 Hard (if-rules)", zorder=3)
    ax.bar(x, soft_v, w, color=BAR["soft"], label="软 Soft (TCN)", zorder=3)
    ax.bar(x + w, fused_v, w, color=BAR["fused"], label="硬加软 Hard+Soft (max)", zorder=3)

    for xi, v in enumerate(hard_v):
        ax.text(xi - w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    for xi, v in enumerate(soft_v):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    for xi, v in enumerate(fused_v):
        ax.text(xi + w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

    # 运动域 if 规则的「全判正常」类别不平衡地板（红虚线）
    for xi, g in enumerate(groups):
        if g == "TCN":
            ax.plot([xi - 0.42, xi + 0.42], [neg_frac[xi], neg_frac[xi]],
                    color="#B2182B", lw=1.0, ls="--", zorder=4)

    # 分组背景 + 分组标题（2 if-else / 1 if-else+FluxEV / 3 TCN）
    spans = [("if-else", 0, 2), ("if-else+FluxEV", 2, 3), ("TCN", 3, 6)]
    for gname, a, b in spans:
        ax.axvspan(a - 0.5, b - 0.5, color=GROUP_COLOR[gname], alpha=0.07, zorder=1)
        ax.text((a + b - 1) / 2, 1.16, GROUP_CN[gname],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=GROUP_COLOR[gname])

    # x 轴标签按组着色
    for xi, g in enumerate(groups):
        ax.get_xticklabels()[xi].set_color(GROUP_COLOR[g])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (acc@0.5)", fontsize=11)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)

    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04),
              ncol=3, fontsize=9.5, handlelength=1.4, columnspacing=1.3)

    ax.set_title("GeoLife (Tsinghua) — 6 behaviours × Hard / Soft / Hard+Soft (local TCN)",
                 fontsize=13, pad=8)

    fig.text(0.01, 0.008,
             "硬 = if 规则 max(逗留15s/聚集/禁区)；软 = 本地 TCN S_dl≥0.5；硬加软 = 逐帧 max(硬,软)。"
             "红虚线 = 运动域 if 规则「全判正常」的类别不平衡地板(非真实识别, AUROC≈0.5)。",
             fontsize=8.5, color="#6E6E6E")

    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save(fig, "fig_六行为_硬软硬加软_本地软.png", outdir=os.path.join(_HERE, "figs"))
    print("saved figs/fig_六行为_硬软硬加软_本地软.png")


if __name__ == "__main__":
    main()
