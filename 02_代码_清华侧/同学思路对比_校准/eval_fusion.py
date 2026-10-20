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
"""硬加软融合 = max(硬 FluxEV 分数, 软 TCN 分数)。

硬 = FluxEV 完整(res["final"])，软 = TCN 逐帧 z-clip（本地 checkpoint tcn_checkpoint.pt）。
逐帧 max 融合，测试段 >= n//2，过滤自然长停留 >120s。

注意：软用本地 CPU checkpoint，与 GPU 跑的软(同学)数值略有差异，融合结果据此近似。
输出：fusion_6behaviors.json
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.CODE_TSINGHUA)
sys.path.insert(0, _PKGP.TCN_DIR)

import torch
import numpy as np
from sklearn import metrics
import geolife_pipeline as gp
import tcn_branch as T
from fusion_ablation import frame_ade, zclip
from eval_hard_accuracy import (inject_geofence, inject_gather, inject_dwell,
                                inject_motion, max_stationary_run)

DATA = T.DEFAULT_DATA
DT = 2.0
HERE = os.path.dirname(os.path.abspath(__file__))

BEHAVIORS = [("Intrusion", "geofence", "rule"),
             ("Gathering", "gather", "rule"),
             ("Lingering", "dwell", "rule"),
             ("Wandering", "wandering", "motion"),
             ("Sprint", "sprint", "motion"),
             ("Zigzag", "zigzag", "motion")]


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
                f1=round(f1, 4), auroc=round(auroc, 4) if auroc is not None else None)


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
    print(f"过滤自然长停留后 = {len(clean)} 条轨迹")

    result = {}
    for label, key, dom in BEHAVIORS:
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

            res = gp.run_pipeline(lat, lon, tg, DT, others=others, zone=zone,
                                  poly=None, q=0.005)
            # 硬 = if 规则（dwell/gather/geofence 取 max，不含 EVT 运动极值）
            hard = np.maximum.reduce([res["dwell"], res["gather"], res["geofence"]])
            x, y = T.latlon_to_local(lat, lon)
            soft = zclip(frame_ade(model, x, y, obs, pred), mu, sigma)  # 软 = TCN
            fused = np.maximum(hard, soft)             # 硬加软 = max(if规则, TCN)

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
        result[key] = dict(label=label, domain=dom, **row)

        h, s, f = row["hard"], row["soft"], row["fused"]
        print(f"{label:<10} 硬 acc={h['acc']:.3f}(A{h['auroc']})  软 acc={s['acc']:.3f}(A{s['auroc']})  "
              f"硬加软 acc={f['acc']:.3f}(A{f['auroc']})  P={f['prec']:.3f} R={f['rec']:.3f} F1={f['f1']:.3f}")

    json.dump(result, open(os.path.join(HERE, "fusion_6behaviors.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nsaved fusion_6behaviors.json")


if __name__ == "__main__":
    main()
