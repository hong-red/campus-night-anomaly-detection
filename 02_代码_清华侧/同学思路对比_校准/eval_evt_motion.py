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
"""EVT（FluxEV-SPOT 运动极值）对后三个运动行为的真实识别率。

行为：Wandering 徘徊 / Sprint 疾跑 / Zigzag 之字形折返
指标：AUROC + acc@0.5（逐帧，测试段 >= n//2，过滤自然长停留 >120s）
纯 EVT 运动分支(res["motion"])，不叠 if 规则、不叠 TCN。

输出：evt_motion_3behaviors.json
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.CODE_TSINGHUA)

import numpy as np
from sklearn import metrics
import geolife_pipeline as gp

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0
HERE = os.path.dirname(os.path.abspath(__file__))


def local_xy(lat, lon):
    lat0, lon0 = lat[0], lon[0]
    x = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110540.0
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / 110540.0
    lon = lon0 + x / (111320.0 * np.cos(np.radians(lat0)))
    return lat, lon


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


def max_stationary_run(speed, dt):
    run = best = 0.0
    for v in speed:
        if v < gp.V_STATIONARY:
            run += dt
            best = max(best, run)
        else:
            run = 0.0
    return best


BEHAVIORS = [("wandering", "Wandering", "loiter"),
             ("sprint", "Sprint", "run"),
             ("zigzag", "Zigzag", "zigzag")]


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
    for key, label, inj in BEHAVIORS:
        g_pool, s_pool = [], []
        for w in clean:
            lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
            n = len(lat)
            if n < int(700 / DT):
                continue
            x, y = local_xy(lat, lon)
            i0 = int(n * 0.55)
            i1 = min(i0 + max(16, int(n * 0.18)), n - 1)
            if inj == "loiter":
                inject_loiter(x, y, i0, i1)
            elif inj == "run":
                inject_run(x, y, i0, i1)
            else:
                inject_zigzag(x, y, i0, i1)
            lat2, lon2 = xy_to_latlon(x, y, lat[0], lon[0])
            gt = np.zeros(n); gt[i0:i1] = 1

            res = gp.run_pipeline(lat2, lon2, tg, DT, others=None, zone=None,
                                  poly=None, q=0.005)
            motion = res["motion"]          # 纯 EVT 运动极值分数
            mask = np.arange(n) >= n // 2   # FluxEV 需前一半训练
            g_pool.append(gt[mask]); s_pool.append(motion[mask])

        g = np.concatenate(g_pool); s = np.concatenate(s_pool)
        pred = (s >= 0.5).astype(int)
        acc = float((pred == g).mean())
        tp = int(((pred == 1) & (g == 1)).sum())
        fp = int(((pred == 1) & (g == 0)).sum())
        fn = int(((pred == 0) & (g == 1)).sum())
        auroc = (metrics.roc_auc_score(g, s)
                 if g.sum() not in (0, len(g)) and np.std(s) > 1e-9 else None)
        neg_frac = float((g == 0).mean())
        result[key] = dict(label=label, acc=round(acc, 4),
                           auroc=round(auroc, 4) if auroc is not None else None,
                           neg_frac=round(neg_frac, 4), tp=tp, fp=fp, fn=fn)
        print(f"{label:<10} acc@0.5={acc:.4f}  AUROC={auroc:.4f}  "
              f"(负例占比 {neg_frac:.3f})  TP={tp} FP={fp} FN={fn}")

    json.dump(result, open(os.path.join(HERE, "evt_motion_3behaviors.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nsaved evt_motion_3behaviors.json")


if __name__ == "__main__":
    main()
