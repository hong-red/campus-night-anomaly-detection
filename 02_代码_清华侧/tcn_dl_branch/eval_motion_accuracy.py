#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FluxEV(EVT/SPOT) 对 徘徊 / 蛇形 / 疾跑 三个运动事件的最终识别率（硬约束：不加 if/else）
=======================================================================================

硬约束
------
识别完全由 FluxEV 的 **EVT(极值理论 / SPOT) 运动分支**给出，逐帧分数 = `run_pipeline` 返回的
`res["motion"]`（速度/加速度/方向变化率三路 FluxEV 取 max），再按项目固定报警阈值 0.5 二值化。
**不叠加任何 if/else 行为规则**（dwell 逗留 / gather 聚集 / geofence 围栏一律不参与），
也不叠加 TCN/DL 分支。

三条运动事件与注入器（与 `fusion_ablation.py` 一致）
----------------------------------------------------
  徘徊 wandering  = inject_loiter  沿行进方向来回折返
  蛇形 zigzag     = inject_zigzag  横向高频摆动
  疾跑 sprint     = inject_run     位移 2.5× 放大(速度突增)

指标（逐帧，dt=2s，注入段占测试段约 36%）
------------------------------------------
  acc@0.5   识别率 = 固定阈值 0.5 的逐帧准确率（与规则域 rule_accuracy.json 的 acc 同口径）
  acc@bestF1 最优 F1 阈值下的逐帧准确率（标量阈值搜索，非 if/else）
  其余：P / R / F1 / AUROC / AP + 混淆矩阵

诊断（EVT 依赖 GPD 拟合，短轨迹会退化）
----------------------------------------
  每条轨迹上报: 点数 n / motion_dims_ok / 测试段 motion 非零占比。
  GPD 需要 train_len(=n/2) 显著大于 ss_idx(=s_w*2+half_d_w+period*(p_w-1))，否则 motion 恒 0。
  故默认用 min_dur=900(≈15min) 的长轨迹保证 GPD 可拟合；可用 --min_dur 放宽观察退化。

输出 (tcn_dl_branch/results/)
-----------------------------
  motion_evt_accuracy.json   三事件全量指标 + 逐轨迹诊断

用法：python eval_motion_accuracy.py [--max_traj 120] [--seed 7] [--min_dur 900]
"""

import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# Windows 下必须先 import torch 再 import numpy（c10.dll 冲突，见 tcn_branch.py）
import torch
import numpy as np
from sklearn import metrics

import tcn_branch as T
import geolife_pipeline as gp

RESULTS = T.RESULTS_DIR

EVENTS = [
    ("wandering", "徘徊 Wandering", T.inject_loiter),
    ("zigzag",    "蛇形 Zigzag",    T.inject_zigzag),
    ("sprint",    "疾跑 Sprint",    T.inject_run),
]


def frame_cm(score, gt, thr=0.5):
    """@固定阈值 0.5 的逐帧混淆矩阵 + acc/P/R/F1。"""
    pred = (np.nan_to_num(score, nan=0.0) >= thr).astype(int)
    gt = gt.astype(int)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    n = len(gt)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(acc=round(acc, 4), prec=round(prec, 4), rec=round(rec, 4),
                f1=round(f1, 4), tp=tp, fp=fp, fn=fn, tn=tn)


def rank_metrics(score, gt):
    s = np.nan_to_num(score, nan=0.0)
    g = gt.astype(int)
    if g.sum() == 0 or g.sum() == len(g) or np.std(s) < 1e-9:
        return {"auroc": None, "ap": None}
    return {"auroc": round(float(metrics.roc_auc_score(g, s)), 4),
            "ap": round(float(metrics.average_precision_score(g, s)), 4)}


def evaluate_event(trajs, injector, dt, q):
    """注入运动异常，逐帧纯 EVT(motion) 打分，池化所有轨迹测试段。"""
    gt_pool, sc_pool = [], []
    diag = []
    for (x, y, lat, lon, tg, _path) in trajs:
        n = len(x)
        if n < 120:                       # 太短无法做 EVT 训练段/测试段划分
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + max(16, int(n * 0.18)), n - 1)
        if i1 - i0 < 8:
            continue
        xa, ya = x.copy(), y.copy()
        injector(xa, ya, i0, i1)
        gt = np.zeros(n)
        gt[i0:i1] = 1
        lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
        res = gp.run_pipeline(lat_a, lon_a, tg, dt, others=None, zone=None,
                              poly=None, q=q)
        motion = res["motion"]            # 纯 EVT 运动分数（不含 if/else 规则）
        mask = np.arange(n) >= n // 2
        gt_pool.append(gt[mask])
        sc_pool.append(motion[mask])
        seg = motion[mask]
        diag.append(dict(n=int(n), dims_ok=int(res["motion_dims_ok"]),
                         degraded=bool(res["motion_degraded"]),
                         nonzero_frac=round(float((seg > 0).mean()), 4)))

    gt = np.concatenate(gt_pool)
    sc = np.concatenate(sc_pool)

    out = {"n_frames": int(len(gt)), "n_pos": int(gt.sum()),
           "n_neg": int((1 - gt).sum()), "n_traj": len(diag),
           "diag": diag,
           "motion_nonzero_frac": round(float((sc > 0).mean()), 4)}
    out["at_0_5"] = frame_cm(sc, gt, 0.5)
    out["at_0_5"].update(rank_metrics(sc, gt))

    # 最优 F1 阈值（标量阈值搜索，非 if/else 规则）
    prec_c, rec_c, thr_c = metrics.precision_recall_curve(gt, sc)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)
    k = int(np.argmax(f1_c))
    best_thr = float(thr_c[k]) if k < len(thr_c) else float(sc.max())
    out["at_best_f1"] = frame_cm(sc, gt, best_thr)
    out["at_best_f1"]["threshold"] = round(best_thr, 4)
    out["at_best_f1"].update(rank_metrics(sc, gt))
    return out


def main():
    ap = argparse.ArgumentParser(description="FluxEV(EVT) 三运动事件识别率（不加 if/else）")
    ap.add_argument("--max_traj", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min_dur", type=float, default=900,
                    help="轨迹最短时长(秒)，默认 900≈15min 保证 EVT/GPD 可拟合")
    ap.add_argument("--dt", type=float, default=2.0)
    args = ap.parse_args()

    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, args.dt, args.max_traj,
                                           args.seed, min_dur=args.min_dur)
    print(f"评估轨迹数 = {len(trajs)}  (dt={args.dt}s, min_dur={args.min_dur}s, "
          f"seed={args.seed})")
    if not trajs:
        print("未找到足够长的行走轨迹，请降低 --min_dur")
        sys.exit(1)
    lens = [len(t[0]) for t in trajs]
    print(f"轨迹点数: min={min(lens)}  med={int(np.median(lens))}  max={max(lens)}\n")

    res = {}
    print("=" * 78)
    print("三运动事件识别率（纯 FluxEV/EVT 运动分支，逐帧，无 if/else 规则）")
    print("=" * 78)
    for key, label, inj in EVENTS:
        r = evaluate_event(trajs, inj, args.dt, q=0.005)
        res[key] = r
        a05, bf = r["at_0_5"], r["at_best_f1"]
        n_deg = sum(1 for d in r["diag"] if d["degraded"])
        print(f"\n[{label}]  n_traj={r['n_traj']} (GPD退化 {n_deg})  "
              f"n_frames={r['n_frames']} (异常 {r['n_pos']}, 正常 {r['n_neg']})  "
              f"motion非零占比={r['motion_nonzero_frac']:.3f}")
        print(f"  识别率 acc@0.5    = {a05['acc']:.4f}   "
              f"(P={a05['prec']:.4f}  R={a05['rec']:.4f}  F1={a05['f1']:.4f}  "
              f"AUROC={a05['auroc']}  AP={a05['ap']})  "
              f"[TP{a05['tp']} FP{a05['fp']} FN{a05['fn']} TN{a05['tn']}]")
        print(f"  识别率 acc@最优F1 = {bf['acc']:.4f}   "
              f"(thr={bf['threshold']:.3f}  P={bf['prec']:.4f}  R={bf['rec']:.4f}  "
              f"F1={bf['f1']:.4f})")

    print("\n" + "-" * 78)
    print("最终识别率（acc@0.5，逐帧，纯 EVT，无 if/else）：")
    for key, label, _ in EVENTS:
        print(f"  {label:<18} {res[key]['at_0_5']['acc']:.4f}")
    print("-" * 78)

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "motion_evt_accuracy.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n已保存 results/motion_evt_accuracy.json")


if __name__ == "__main__":
    main()
