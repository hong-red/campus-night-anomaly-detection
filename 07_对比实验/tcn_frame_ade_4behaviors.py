#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""帧级 TCN 四运动行为表（与主表 eval_and_plot_final_3cols.py 软分支完全同口径）。

AUROC : 对逐帧 ADE 做 nan->mu 填充后取 roc_auc_score(gt, ADE)（ADE 越大越异常），
        与主表 soft 分支的 zclip 分数同序（clip 为单调变换，AUROC 不变）。
ADE   : 真实逐帧 ADE(m) 的均值，仅统计非 NaN 帧。
轨迹集: 与主表同一批 50 条 clean Geolife 行走轨迹。
"""
import os, sys, json
ROOT = r"F:\双分支实验_完整交付"
sys.path.insert(0, ROOT)
import _paths as PK
for p in (PK.CODE_TSINGHUA, os.path.join(PK.CODE_TSINGHUA, "tcn_dl_branch")):
    sys.path.insert(0, p)
import torch
import numpy as np
from sklearn import metrics
import geolife_pipeline as gp
import tcn_branch as T

DT = 2.0
DATA = PK.GEOLIFE_DATA
OUT = r"F:\双分支实验_完整交付\07_对比实验\结果_JSON\tcn_frame_ade_4behaviors.json"


def frame_ade(model, x, y, obs_len, pred_len):
    n = len(x); total = obs_len + pred_len
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


def max_stationary_run(speed, dt):
    run = best = 0.0
    for v in speed:
        if v < gp.V_STATIONARY:
            run += dt; best = max(best, run)
        else:
            run = 0.0
    return best


EVENTS = [
    ("sprint",      "Sprinting",   "疾跑", T.inject_run),
    ("wandering",   "Loitering",   "徘徊", T.inject_loiter),
    ("zigzag",      "Zigzag",      "蛇形", T.inject_zigzag),
    ("sudden-stop", "Sudden stop", "急停", T.inject_stop),
]


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    print(f"TCN obs={obs} pred={pred} mu={mu:.2f} sigma={sigma:.2f}")

    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    clean = []
    for w in walks:
        lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
        speed0, _, _ = gp.trajectory_features(lat, lon, DT)
        if max_stationary_run(speed0, DT) <= 120:
            clean.append(w)
    print(f"clean trajs = {len(clean)}")

    out = {}
    for key, en, cn, inj in EVENTS:
        g_all, s_all = [], []          # 打分用（nan->mu），匹配主表
        gv_all, ade_all = [], []       # ADE 均值用（仅非 NaN）
        used = 0
        for w in clean:
            lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
            n = len(lat)
            if n < int(700 / DT):
                continue
            i0 = int(n * 0.55); i1 = min(i0 + max(16, int(n * 0.18)), n - 1)
            x, y = T.latlon_to_local(lat, lon)
            inj(x, y, i0, i1)
            gt = np.zeros(n); gt[i0:i1] = 1
            ade = frame_ade(model, x, y, obs, pred)
            m = np.arange(n) >= n // 2
            g = gt[m]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            score = np.nan_to_num(ade[m], nan=mu)   # 主表 zclip 同序
            g_all.append(g); s_all.append(score)
            ok = ~np.isnan(ade[m])
            gv_all.append(g[ok]); ade_all.append(ade[m][ok])
            used += 1
        g = np.concatenate(g_all); s = np.concatenate(s_all)
        gv = np.concatenate(gv_all); av = np.concatenate(ade_all)
        auroc = float(metrics.roc_auc_score(g, s))
        ade_anom = float(np.mean(av[gv == 1]))
        ade_norm = float(np.mean(av[gv == 0]))
        out[key] = dict(label_en=en, label_cn=cn, level="frame",
                        auroc=round(auroc, 4),
                        ade_anomaly=round(ade_anom, 2), ade_normal=round(ade_norm, 2),
                        n_frames=int(len(s)), n_pos=int(g.sum()), n_neg=int((1 - g).sum()),
                        n_traj=used)
        print(f"{cn} {en:<12} AUROC={auroc:.4f}  ADE_anom={ade_anom:.2f}  "
              f"ADE_norm={ade_norm:.2f}  frames={len(s)} pos={int(g.sum())} neg={int((1-g).sum())}")

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
