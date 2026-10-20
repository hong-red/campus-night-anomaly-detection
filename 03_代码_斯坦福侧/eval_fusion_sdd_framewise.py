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
"""SDD 帧级 6 行为 × 3 检测器（硬 if规则 / 软 S_dl / 硬加软 max）准确率。

对应 F:\\仿真跑\\同学思路对比_校准\\eval_fusion.py 的斯坦福版。
口径：帧级（逐帧、整条轨迹、多轨迹池化），阈值 0.5。
  硬     = max(dwell, gather, geofence) if 规则
  软     = zclip S_dl（SDD 重训 TCN）
  硬加软 = max(硬, 软)
输出：results/sdd_fusion_framewise.json
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, _PKGP.CODE_TSINGHUA)
sys.path.insert(0, _PKGP.TCN_DIR)

import torch                      # Windows: torch 必须先于 numpy (c10.dll)
import numpy as np
from sklearn import metrics

import sdd_common as S
from fusion_ablation_sdd import inject_event

BEHAVIORS = [("Intrusion", "geofence"), ("Gathering", "gather"), ("Lingering", "dwell"),
             ("Wandering", "wandering"), ("Sprint", "sprint"), ("Zigzag", "zigzag")]


def rules_dl_fused(model, stats, traj, event, obs, pred):
    """注入事件，返回 (gt, {rules, dl, fused}) 逐帧分数（整条轨迹）。"""
    sc, x, y, t = traj
    mu, sigma = stats["mu"], stats["sigma"]
    xa, ya, gt, others, zone = inject_event(x, y, event, 1.0)
    speed, _accel, _turn = S.run_sdd.features_2d(xa, ya, 1.0)
    dwell = S.stationary_episode(speed, 1.0)
    gather = S.gather_score_2d(xa, ya, others, 1.0) if others else np.zeros(len(xa))
    geof = S.geofence_score_2d(xa, ya, zone) if zone else np.zeros(len(xa))
    rules = np.maximum.reduce([dwell, gather, geof])
    dl = S.zclip(S.frame_ade(model, xa, ya, obs, pred), mu, sigma)
    fused = np.maximum(rules, dl)
    return gt, {"rules": rules, "dl": dl, "fused": fused}


def pool_metrics(g_pool, s_pool):
    g = np.concatenate(g_pool)
    s = np.concatenate(s_pool)
    pred = (s >= 0.5).astype(int)
    acc = float((pred == g).mean())
    tp = int(((pred == 1) & (g == 1)).sum())
    fp = int(((pred == 1) & (g == 0)).sum())
    fn = int(((pred == 0) & (g == 1)).sum())
    tn = int(((pred == 0) & (g == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auroc = (metrics.roc_auc_score(g, s)
             if g.sum() not in (0, len(g)) and np.std(s) > 1e-9 else None)
    return dict(acc=round(acc, 4), prec=round(prec, 4), rec=round(rec, 4),
                f1=round(f1, 4), auroc=round(auroc, 4) if auroc is not None else None,
                n=int(len(g)), pos=round(float(g.mean()), 4),
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    model, stats, cfg = S.ensure_sdd_tcn()
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    print(f"TCN: obs={obs} pred={pred} mu={stats['mu']:.2f}m sigma={stats['sigma']:.2f}m")
    trajs = S.collect_sdd_walking(1.0, 150, seed=7, min_len=60)
    print(f"评估行走轨迹数 = {len(trajs)}")

    result = {}
    for label, event in BEHAVIORS:
        pools = {d: ([], []) for d in ("rules", "dl", "fused")}
        for tr in trajs:
            if len(tr[1]) < 60:
                continue
            gt, scs = rules_dl_fused(model, stats, tr, event, obs, pred)
            if gt.sum() == 0 or gt.sum() == len(gt):
                continue
            for d in ("rules", "dl", "fused"):
                pools[d][0].append(gt)
                pools[d][1].append(scs[d])
        row = {d: pool_metrics(pools[d][0], pools[d][1]) for d in ("rules", "dl", "fused")}
        result[event] = dict(label=label, **row)
        h, s, f = row["rules"], row["dl"], row["fused"]
        print(f"{label:<10} 硬 acc={h['acc']:.3f}(A{h['auroc']})  "
              f"软 acc={s['acc']:.3f}(A{s['auroc']})  "
              f"硬加软 acc={f['acc']:.3f}(A{f['auroc']})  "
              f"[P={f['prec']:.3f} R={f['rec']:.3f} F1={f['f1']:.3f}]")

    out = os.path.join(S.RESULTS_DIR, "sdd_fusion_framewise.json")
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
