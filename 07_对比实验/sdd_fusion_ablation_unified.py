# -*- coding: utf-8 -*-
"""七事件 × 五检测器消融表（斯坦福 SDD，micro + zclip，与 §一 主表同口径）。

口径：collect_sdd_walking(1.0, 150, seed=7, min_len=60)（51 条）；micro（整条轨迹拼接）；
TCN = z-clip。SDD 侧 dwell 全用 lo=15s（DWELL_LO=15），故仅if规则 = FluxEV（无 EVT 运动异常时两列相同）。

输出：07_对比实验/结果_JSON/sdd_fusion_ablation_unified.json
"""
import os
import sys
import json

ROOT = r"F:\双分支实验_完整交付"
sys.path.insert(0, ROOT)
import _paths as PK
for p in (PK.CODE_STANFORD, PK.CODE_TSINGHUA, os.path.join(PK.CODE_TSINGHUA, "tcn_dl_branch")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import numpy as np
from sklearn import metrics
import sdd_common as S
from fusion_ablation_sdd import inject_event

OUT = os.path.join(ROOT, "07_对比实验", "结果_JSON", "sdd_fusion_ablation_unified.json")
DETECTORS = ["rules", "fluxev", "dl", "naive", "gated"]
EVENTS = ["geofence", "gather", "dwell", "wandering", "sprint", "zigzag", "sudden-stop"]
EVENT_CN = {"geofence": "闯入", "gather": "聚集", "dwell": "逗留",
            "wandering": "徘徊", "sprint": "疾跑", "zigzag": "蛇形", "sudden-stop": "急停"}


def score_event_unified(model, stats, traj, event, obs, pred):
    sc, x, y, t = traj
    mu, sigma = stats["mu"], stats["sigma"]
    xa, ya, gt, others, zone = inject_event(x, y, event, 1.0)

    speed, _accel, _turn = S.run_sdd.features_2d(xa, ya, 1.0)
    dwell = S.stationary_episode(speed, 1.0)
    gather = S.gather_score_2d(xa, ya, others, 1.0) if others else np.zeros(len(xa))
    geof = S.geofence_score_2d(xa, ya, zone) if zone else np.zeros(len(xa))
    rules = np.maximum.reduce([dwell, gather, geof])

    dl = S.zclip(S.frame_ade(model, xa, ya, obs, pred), mu, sigma)
    res = S.run_sdd_pipeline(xa, ya, t, 1.0, others=others, zone=zone)
    fluxev = res["final"]
    naive = np.maximum(fluxev, dl)
    gated = np.maximum(fluxev, dl * (dl >= 0.5))
    return gt, {"rules": rules, "fluxev": fluxev, "dl": dl, "naive": naive, "gated": gated}


def micro_auroc(g, s):
    auroc = (metrics.roc_auc_score(g, s)
             if g.sum() not in (0, len(g)) and np.std(s) > 1e-9 else None)
    return round(auroc, 4) if auroc is not None else None


def main():
    model, stats, cfg = S.ensure_sdd_tcn()
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    print(f"TCN obs={obs} pred={pred} mu={stats['mu']:.2f} sigma={stats['sigma']:.2f}")
    trajs = S.collect_sdd_walking(1.0, 150, seed=7, min_len=60)
    print(f"评估行走轨迹数 = {len(trajs)}")

    out = {}
    for event in EVENTS:
        pools = {d: ([], []) for d in DETECTORS}
        used = 0
        for tr in trajs:
            if len(tr[1]) < 60:
                continue
            gt, scs = score_event_unified(model, stats, tr, event, obs, pred)
            if gt.sum() == 0 or gt.sum() == len(gt):
                continue
            for d in DETECTORS:
                pools[d][0].append(gt); pools[d][1].append(scs[d])
            used += 1
        out[event] = {d: micro_auroc(np.concatenate(pools[d][0]), np.concatenate(pools[d][1]))
                      for d in DETECTORS}
        print(f"{EVENT_CN[event]:<4} traj={used:>3}  rules={out[event]['rules']} "
              f"fluxev={out[event]['fluxev']} dl={out[event]['dl']} "
              f"naive={out[event]['naive']} gated={out[event]['gated']}")

    json.dump(dict(_meta=dict(level="frame", aggregation="micro", tcn_score="z-clip", dt=1.0,
                              traj="collect_sdd_walking(1.0,150,seed=7,min_len=60)"),
                   **out), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
