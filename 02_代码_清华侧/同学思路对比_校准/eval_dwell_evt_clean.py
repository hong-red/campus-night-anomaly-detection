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
"""干净消融：原地逗留（仅 dwell，不注入 gather/geofence）——
「dwell 时间规则」 vs 「EVT 运动极值」 vs 「两者 max」

回答同学的问题：原地逗留加 EVT 运动极值，AUROC 到底是升还是降。
与 fusion_ablation.py 的差别：这里只注入 dwell 一个事件，消除了 gather/geofence
的跨事件混淆，且逐轨迹池化（不只用一条超长轨迹）。
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
N_TRAJ = 15
DWELL_SEC = 360

def auroc(g, s):
    if g.sum() == 0 or g.sum() == len(g) or np.std(s) < 1e-9:
        return None
    return float(metrics.roc_auc_score(g, s))

def main():
    walks = gp.scan_walking_trajectories(DATA, max_files=300, min_dur=900)
    print(f"候选长轨迹 {len(walks)} 条，取前 {N_TRAJ} 条注入 dwell ...")
    agg = {"dwell_rule": [], "motion": [], "max_rule_motion": []}
    used = 0
    for (npts, med, path, lat5, lon5, tg5) in walks:
        lat, lon, tg = gp.resample(lat5, lon5, tg5, DT)
        n = len(lat)
        if n < 300:
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + int(DWELL_SEC / DT), n - 2)
        lat = lat.copy(); lon = lon.copy()
        lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
        gt = np.zeros(n); gt[i0:i1] = 1

        res = gp.run_pipeline(lat, lon, tg, DT, others=None, zone=None,
                              poly=None, q=0.005)
        dwell = res["dwell"]                    # dwell 时间阈值规则
        motion = res["motion"]                  # EVT 运动极值(速度/加速度/方向)
        mx = np.maximum(dwell, motion)

        mask = np.arange(n) >= n // 2           # 只看测试段(训练段之后)
        g = gt[mask]
        if g.sum() == 0 or g.sum() == len(g):
            continue
        for name, s in (("dwell_rule", dwell), ("motion", motion),
                        ("max_rule_motion", mx)):
            a = auroc(g, s[mask])
            if a is not None:
                agg[name].append(a)
        used += 1
        if used >= N_TRAJ:
            break

    print(f"\n已用轨迹数 = {used}")
    print(f"{'检测器':<16}{'AUROC(均值)':<12}{'最小':<8}{'最大':<8}{'条数'}")
    for name in ("dwell_rule", "motion", "max_rule_motion"):
        v = agg[name]
        if not v:
            print(f"{name:<16}  (无数据)")
            continue
        print(f"{name:<16}{np.mean(v):<12.3f}{np.min(v):<8.3f}{np.max(v):<8.3f}{len(v)}")

    a = np.array(agg["dwell_rule"]); b = np.array(agg["max_rule_motion"])
    n = min(len(a), len(b))
    if n:
        up = int((b[:n] > a[:n] + 1e-6).sum())
        down = int((b[:n] < a[:n] - 1e-6).sum())
        same = n - up - down
        print(f"\n加 EVT 运动极值后：升 {up} 条 / 降 {down} 条 / 持平 {same} 条  "
              f"(均值 {np.mean(a[:n]):.3f} -> {np.mean(b[:n]):.3f})")

    out = {k: round(float(np.mean(v)), 4) for k, v in agg.items() if v}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "dwell_evt_clean.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nsaved dwell_evt_clean.json ->", out)

if __name__ == "__main__":
    main()
