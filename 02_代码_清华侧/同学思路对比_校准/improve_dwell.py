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
"""提升「原地逗留」检测的几个变体对比

基线：dwell 规则(120/300s) 0.831 / acc 0.778，加运动极值后掉到 0.771 / 0.734。
根因：运动极值的 1381 个误报全在正常行走段，真贡献只在起停瞬间(仅 26 TP)；
      dwell 规则唯一短板是 120s 起步延迟(漏掉每段逗留前 120s)。

变体：
  A. 缩短 dwell 起步延迟 lo ∈ {120,60,30,15}（更快报警，代价是可能误报正常短暂驻足）
  B. 运动极值加"仅静止(speed<0.4)才放行"的门，砍掉正常行走误报
  C. A+B 组合

指标：AUROC(逐轨迹均值) + 逐帧 acc@0.5(池化)
"""

import os
import sys
import json

sys.path.insert(0, _PKGP.CODE_TSINGHUA)
sys.path.insert(0, _PKGP.STYLE_DIR)

import numpy as np
from sklearn import metrics
import geolife_pipeline as gp

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0
DWELL_SEC = 360
VS = gp.V_STATIONARY
HERE = os.path.dirname(os.path.abspath(__file__))


def auroc(g, s):
    if g.sum() == 0 or g.sum() == len(g) or np.std(s) < 1e-9:
        return None
    return float(metrics.roc_auc_score(g, s))


def dwell(speed, lo, hi):
    return gp.stationary_episode_score(speed, DT, lo, hi)


def variants(speed, motion):
    d120 = dwell(speed, 120, 300)
    return {
        "dwell 120/300":        d120,
        "dwell 60/300":         dwell(speed, 60, 300),
        "dwell 30/300":         dwell(speed, 30, 300),
        "dwell 15/300":         dwell(speed, 15, 300),
        "dwell120 + 运动":       np.maximum(d120, motion),
        "dwell120 + 运动(限静止)": np.maximum(d120, motion * (speed < VS)),
        "dwell60 + 运动(限静止)":  np.maximum(dwell(speed, 60, 300), motion * (speed < VS)),
        "dwell30 + 运动(限静止)":  np.maximum(dwell(speed, 30, 300), motion * (speed < VS)),
    }


def main():
    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    print(f"长轨迹 = {len(walks)} 条")

    names = None
    pool_gt = []
    pool = None
    au_agg = None

    for (npts, med, path, lat5, lon5, tg5) in walks:
        lat, lon, tg = gp.resample(lat5, lon5, tg5, DT)
        n = len(lat)
        if n < int(700 / DT):
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + int(DWELL_SEC / DT), n - 2)
        lat = lat.copy(); lon = lon.copy()
        lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
        gt = np.zeros(n); gt[i0:i1] = 1

        speed, _, _ = gp.trajectory_features(lat, lon, DT)
        res = gp.run_pipeline(lat, lon, tg, DT, others=None, zone=None,
                              poly=None, q=0.005)
        motion = res["motion"]

        mask = np.arange(n) >= n // 2
        g = gt[mask]
        if g.sum() == 0 or g.sum() == len(g):
            continue

        vs = variants(speed, motion)
        if names is None:
            names = list(vs.keys())
            pool = {k: [] for k in names}
            au_agg = {k: [] for k in names}
        pool_gt.append(g)
        for k, s in vs.items():
            pool[k].append(s[mask])
            a = auroc(g, s[mask])
            if a is not None:
                au_agg[k].append(a)

    g_all = np.concatenate(pool_gt)
    print(f"\n{'变体':<24}{'AUROC':<8}{'acc@0.5':<9}{'TP':<6}{'FP':<6}{'FN':<6}")
    rows = {}
    for k in names:
        s_all = np.concatenate(pool[k])
        pred = (s_all >= 0.5).astype(int)
        acc = float((pred == g_all).mean())
        tp = int(((pred == 1) & (g_all == 1)).sum())
        fp = int(((pred == 1) & (g_all == 0)).sum())
        fn = int(((pred == 0) & (g_all == 1)).sum())
        au = float(np.mean(au_agg[k]))
        print(f"{k:<24}{au:<8.3f}{acc:<9.4f}{tp:<6}{fp:<6}{fn:<6}")
        rows[k] = dict(auroc=round(au, 4), acc_at_0_5=round(acc, 4),
                       tp=tp, fp=fp, fn=fn)

    json.dump(rows, open(os.path.join(HERE, "improve_dwell.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nsaved improve_dwell.json")


if __name__ == "__main__":
    main()
