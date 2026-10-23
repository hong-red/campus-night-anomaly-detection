#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨校交叉验证 —— 帧级版（与主表同口径）。

在 cross_validate_schools.py 的窗口级实现基础上，把 window_auroc 换成
frame_ade（逐帧 ADE，参考 tcn_dl_branch/fusion_ablation.py 的 frame_ade），
其余（模型、测试轨迹、注入函数、注入位置）完全沿用，保证只改评估粒度。
"""
import os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
for _p in (_PROJ, os.path.join(_PROJ, "02_代码_清华侧"),
           os.path.join(_PROJ, "02_代码_清华侧", "tcn_dl_branch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from sklearn import metrics
import sdd_common as S
import tcn_branch as T

RESULTS = S.RESULTS_DIR
GEO_TCN_DT1 = os.path.join(RESULTS, "geo_tcn_dt1.pt")
BEHAVIORS = [("wandering", "Wandering", T.inject_loiter),
             ("loitering", "Loitering (in place)", T.inject_stop),
             ("sprint", "Sprint", T.inject_run),
             ("zigzag", "Zigzag", T.inject_zigzag)]
OBS, PRED = 16, 8


def frame_ade(model, x, y, obs_len, pred_len):
    """整条轨迹逐帧 ADE(米)；无窗口处 NaN。"""
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


def frame_auroc(model, tracks_xy, inject_fn):
    """逐帧 AUROC：注入段 = 正样本，其余 = 负样本（与注入位置一致）。"""
    scores, labels = [], []
    for (x, y) in tracks_xy:
        n = len(x)
        if n < OBS + PRED + 24:
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + max(16, int(n * 0.18)), n - OBS - PRED - 2)
        if i1 - i0 < 8:
            continue
        xa, ya = x.copy(), y.copy()
        inject_fn(xa, ya, i0, i1)
        ade = frame_ade(model, xa, ya, OBS, PRED)
        lab = np.zeros(n)
        lab[i0:i1] = 1
        ok = ~np.isnan(ade)
        scores.append(ade[ok]); labels.append(lab[ok])
    if not scores:
        return float("nan")
    s = np.concatenate(scores); l = np.concatenate(labels)
    if l.sum() in (0, len(l)):
        return float("nan")
    return float(metrics.roc_auc_score(l, s))


def main():
    sdd_model, sdd_stats, sdd_cfg = S.ensure_sdd_tcn()
    if os.path.exists(GEO_TCN_DT1):
        geo_model, geo_stats, geo_cfg = T.load_predictor(GEO_TCN_DT1)
    else:
        raise SystemExit("缺少 Geolife dt=1s checkpoint：先跑 cross_validate_schools.py")

    sdd_test = [(x, y) for _, x, y, _ in S.collect_sdd_walking(1.0, 100, seed=42, min_len=48)]
    geo_test = [(x, y) for x, y, *_ in T.collect_walking_trajectories(T.DEFAULT_DATA, 1.0, 100, seed=42)]
    print(f"测试轨迹: Stanford {len(sdd_test)} / Geolife {len(geo_test)}")

    combos = [("geo_sdd", "Geo→SDD", geo_model, sdd_test),
              ("sdd_geo", "SDD→Geo", sdd_model, geo_test),
              ("geo_geo", "Geo→Geo", geo_model, geo_test),
              ("sdd_sdd", "SDD→SDD", sdd_model, sdd_test)]

    out = {b[0]: {} for b in BEHAVIORS}
    for key, en, inj in BEHAVIORS:
        for ck, cname, model, tracks in combos:
            out[key][ck] = round(frame_auroc(model, tracks, inj), 3)
            print(f"  {en:<22}{cname:<10}{out[key][ck]:.3f}")

    path = os.path.join(RESULTS, "cross_validate_schools_frame.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已保存:", path)


if __name__ == "__main__":
    main()
