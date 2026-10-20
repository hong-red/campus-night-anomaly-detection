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
"""原地逗留准确率 → ~90% 需要什么配置？扫阈值给出答案。

dwell 规则分数 = clip((run - lo)/(300 - lo), 0, 1)，run=当前已停留秒数。
@0.5 报警点 = run=(lo+300)/2。要更高准确率就要把报警点前移（阈值更低 / lo 更小）。
本脚本扫阈值，报告：acc@0.5、达到 90% 准确率所需的阈值与报警点、以及最大 acc。
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
DWELL_SEC = 360
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    print(f"长轨迹 = {len(walks)} 条")

    pools = {lo: {"g": [], "s": []} for lo in (15, 30, 60)}
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
        mask = np.arange(n) >= n // 2
        g = gt[mask]
        if g.sum() == 0 or g.sum() == len(g):
            continue
        for lo in pools:
            s = gp.stationary_episode_score(speed, DT, lo, 300)
            pools[lo]["g"].append(g)
            pools[lo]["s"].append(s[mask])

    for lo in (15, 30, 60):
        g = np.concatenate(pools[lo]["g"])
        s = np.concatenate(pools[lo]["s"])
        npos = int(g.sum()); nneg = int((1 - g).sum())

        def stats(thr):
            pred = (s >= thr).astype(int)
            acc = float((pred == g).mean())
            tp = int(((pred == 1) & (g == 1)).sum())
            fp = int(((pred == 1) & (g == 0)).sum())
            return acc, tp, fp

        acc05, tp05, fp05 = stats(0.5)
        # 找达到 90% 准确率的最小阈值(报警点最晚)
        acc90 = fp90 = thr90 = None
        best_acc = best_thr = best_fp = -1
        for thr in np.arange(0.02, 1.0, 0.01):
            acc, tp, fp = stats(thr)
            if acc > best_acc:
                best_acc, best_thr, best_fp = acc, float(thr), fp
            if acc >= 0.90 and acc90 is None:
                acc90, fp90, thr90 = acc, fp, float(thr)
        run90 = lo + thr90 * (300 - lo) if thr90 is not None else None
        run_best = lo + best_thr * (300 - lo)

        print(f"\n=== dwell lo={lo}s / hi=300s ===  (正例 {npos} 帧, 负例 {nneg} 帧)")
        print(f"  @0.5     acc={acc05:.4f}  TP={tp05}  FP={fp05}  (报警点={lo + 0.5*(300-lo):.0f}s)")
        if thr90 is not None:
            print(f"  达90%    acc={acc90:.4f}  阈值={thr90:.2f}  FP={fp90}  (报警点≈{run90:.0f}s)")
        else:
            print("  达90%    无法达到(正例占比有限)")
        print(f"  最大acc  acc={best_acc:.4f}  阈值={best_thr:.2f}  FP={best_fp}  (报警点≈{run_best:.0f}s)")


if __name__ == "__main__":
    main()
