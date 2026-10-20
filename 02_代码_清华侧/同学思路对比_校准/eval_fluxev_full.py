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
"""「硬」= FluxEV 完整硬性分支 = max(EVT 运动极值, if 规则) = res["final"]。

按同学框架：第一个分支 = 规则(if) + EVT 极值 取 max。这里把「硬」算成完整 FluxEV，
给出 6 行为的准确率 acc@0.5 与 AUROC（逐帧，测试段>=n//2，过滤自然长停留>120s）。

输出：fluxev_full_6behaviors.json
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.CODE_TSINGHUA)

import numpy as np
from sklearn import metrics
import geolife_pipeline as gp
from eval_hard_accuracy import (inject_geofence, inject_gather, inject_dwell,
                                inject_motion, max_stationary_run)

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0
HERE = os.path.dirname(os.path.abspath(__file__))

BEHAVIORS = [("Intrusion", "geofence", "rule"),
             ("Gathering", "gather", "rule"),
             ("Lingering", "dwell", "rule"),
             ("Wandering", "wandering", "motion"),
             ("Sprint", "sprint", "motion"),
             ("Zigzag", "zigzag", "motion")]


def main():
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
        g_pool, s_pool = [], []
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
            final = res["final"]              # 硬 = FluxEV 完整 = max(EVT运动, 规则)
            mask = np.arange(n) >= n // 2
            g_pool.append(gt[mask]); s_pool.append(final[mask])

        g = np.concatenate(g_pool); s = np.concatenate(s_pool)
        pred = (s >= 0.5).astype(int)
        acc = float((pred == g).mean())
        tp = int(((pred == 1) & (g == 1)).sum())
        fp = int(((pred == 1) & (g == 0)).sum())
        fn = int(((pred == 0) & (g == 1)).sum())
        auroc = (metrics.roc_auc_score(g, s)
                 if g.sum() not in (0, len(g)) and np.std(s) > 1e-9 else None)
        neg_frac = float((g == 0).mean())
        result[key] = dict(label=label, domain=dom, acc=round(acc, 4),
                           auroc=round(auroc, 4) if auroc is not None else None,
                           neg_frac=round(neg_frac, 4), tp=tp, fp=fp, fn=fn)
        print(f"{label:<10} acc@0.5={acc:.4f}  AUROC={auroc:.4f}  "
              f"(负例占比 {neg_frac:.3f})  TP={tp} FP={fp} FN={fn}")

    json.dump(result, open(os.path.join(HERE, "fluxev_full_6behaviors.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nsaved fluxev_full_6behaviors.json")


if __name__ == "__main__":
    main()
