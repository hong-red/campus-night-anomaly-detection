#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算规则域三项(逗留/围栏/聚集)的指标 + 总体指标。

口径（相比旧版的两处修正）
--------------------------
1. 每个事件用它【自己的规则分数】打分，而不是 max(dwell, gather, geofence) 三路合并。
   旧版用合并分数导致：gather/geofence 在正确触发时被记成逗留的"误报"，
   把逗留 acc 从 0.969 拉到 0.898、P 从 1.0 拉到 0.24。
2. 除 acc/P/R/F1@0.5 外，同时给 AUROC / AP（阈值无关，不受逗留"线性抬升 ramp"影响）。
   acc 在负样本占 94.6% 的数据里没有区分度，应优先看 AUROC/AP 和 F1。

输出 (tcn_dl_branch/results/rule_accuracy.json)
-----------------------------------------------
  per_event[kind].specific_rule  —— 该事件自己的规则（论文主数字）
  per_event[kind].full_system    —— 完整融合系统 max(FluxEV, TCN)，一对多口径，仅参考
  overall_union                  —— 完整系统对规则域并集 GT 的总体
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch
from sklearn import metrics
import tcn_branch as T
import geolife_pipeline as gp
import fusion_ablation as FA


def cm_metrics(score, g, thr=0.5):
    """@固定阈值 0.5 的混淆矩阵 + acc/P/R/F1。"""
    pred = (np.nan_to_num(score) >= thr).astype(int)
    g = g.astype(int)
    tp = int(((pred == 1) & (g == 1)).sum())
    fp = int(((pred == 1) & (g == 0)).sum())
    fn = int(((pred == 0) & (g == 1)).sum())
    tn = int(((pred == 0) & (g == 0)).sum())
    acc = (tp + tn) / len(g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(acc=round(acc, 4), prec=round(prec, 4), rec=round(rec, 4),
                f1=round(f1, 4), tp=tp, fp=fp, fn=fn, tn=tn)


def rank_metrics(score, g):
    """阈值无关的 AUROC / AP（连续分数排序能力）。"""
    s = np.nan_to_num(score)
    g = g.astype(int)
    if g.sum() == 0 or g.sum() == len(g) or np.std(s) < 1e-9:
        return {"auroc": None, "ap": None}
    return {"auroc": round(float(metrics.roc_auc_score(g, s)), 4),
            "ap": round(float(metrics.average_precision_score(g, s)), 4)}


def full_metrics(score, g):
    """cm + rank 合并成一个 dict。"""
    d = cm_metrics(score, g)
    d.update(rank_metrics(score, g))
    return d


def main():
    model, stats, cfg = T.load_predictor(FA.CKPT)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]

    walks = gp.scan_walking_trajectories(T.DEFAULT_DATA, max_files=2000, min_dur=900)
    npts, med, path, lat5, lon5, tg5 = walks[0]
    lat, lon, tg = gp.resample(lat5, lon5, tg5, 2.0)
    lat, lon, t, gt, others, zone = FA.build_rule_scenario(lat, lon, tg, 2.0)

    res = gp.run_pipeline(lat, lon, t, 2.0, others=others, zone=zone, q=0.005)
    fluxev = res["final"]
    x, y = T.latlon_to_local(lat, lon)
    dl = FA.zclip(FA.frame_ade(model, x, y, obs, pred), mu, sigma)
    full = np.maximum(fluxev, dl)

    n = len(lat)
    test_start = n // 2
    mask = np.arange(n) >= test_start
    union = np.maximum.reduce([gt["dwell"], gt["gather"], gt["geofence"]])[mask]

    print("=" * 92)
    print(f"轨迹: {os.path.basename(path)}  (测试段 {mask.sum()} 帧, 异常帧 {union.sum()}, "
          f"占比 {union.mean()*100:.1f}%)")
    print("口径: 每事件用【自己的规则分数】打分 + 阈值无关的 AUROC/AP")
    print("=" * 92)

    per_event = {}
    for kind, name in (("dwell", "长时间逗留"), ("geofence", "禁区闯入"),
                       ("gather", "多人聚集")):
        g = gt[kind][mask]
        own = full_metrics(res[kind][mask], g)
        f = full_metrics(full[mask], g)
        per_event[kind] = {"specific_rule": own, "full_system": f}
        print(f"\n[{name}]  GT段 {g.sum()} 帧")
        print(f"  自己规则:   acc={own['acc']:.4f}  P={own['prec']:.4f}  R={own['rec']:.4f}  "
              f"F1={own['f1']:.4f}  AUROC={own['auroc']}  AP={own['ap']}"
              f"   [TP{own['tp']} FP{own['fp']} FN{own['fn']} TN{own['tn']}]")
        print(f"  完整系统:   acc={f['acc']:.4f}  P={f['prec']:.4f}  R={f['rec']:.4f}  "
              f"F1={f['f1']:.4f}  AUROC={f['auroc']}  AP={f['ap']}"
              f"   [TP{f['tp']} FP{f['fp']} FN{f['fn']} TN{f['tn']}]  (一对多, 仅参考)")

    # 逗留的检测延迟（解释 R 为何不是 1.0：线性抬升 ramp 的阈值假象）
    d0 = test_start + int(150 / 2.0)
    d1 = d0 + int(360 / 2.0)
    dwell_seg = res["dwell"][d0:d1]
    hit = np.where(dwell_seg >= 0.5)[0]
    delay = (int(hit[0]) * 2.0) if len(hit) else None
    print(f"\n[逗留检测延迟] 注入 360s 停留, 分数 >=0.5 出现在停留开始后 {delay}s "
          f"(规则语义: 满 {gp.DWELL_MIN}s 才到 1.0, @0.5 需 {(gp.DWELL_MIN+gp.DWELL_LOW)/2:.0f}s)")
    print(f"  -> 逗留 R={per_event['dwell']['specific_rule']['rec']:.4f} 不是漏检, 是 ramp+阈值的必然: "
          f"前 {(gp.DWELL_MIN+gp.DWELL_LOW)/2:.0f}s 被判负, 360s 里后 {360-(gp.DWELL_MIN+gp.DWELL_LOW)/2:.0f}s 判正")

    # 规则域三项平均准确率（各自规则）
    avg_rule = float(np.mean([per_event[k]["specific_rule"]["acc"] for k in per_event]))
    print("\n" + "-" * 92)
    print(f"规则域三项平均准确率(各自规则): {avg_rule:.4f}")

    # 总体（union GT，完整系统）
    tot = full_metrics(full[mask], union)
    out = {"per_event": per_event, "overall_union": {"full_system": tot}}
    out["specific_rule_avg_acc"] = round(avg_rule, 4)
    out["note"] = ("per_event[kind].specific_rule 用该事件自己的规则分数；"
                   "full_system 用 max(FluxEV, TCN) 在单事件上是一对多口径，"
                   "会把其它事件的报警计入该事件 FP，仅作参考；overall_union 是完整系统对并集。")

    print(f"\n总体(规则域并集GT, 完整系统): acc={tot['acc']:.4f}  P={tot['prec']:.4f}  "
          f"R={tot['rec']:.4f}  F1={tot['f1']:.4f}  AUROC={tot['auroc']}  AP={tot['ap']}"
          f"  [TP{tot['tp']} FP{tot['fp']} FN{tot['fn']} TN{tot['tn']}]")

    import json
    json.dump(out, open(os.path.join(FA.RESULTS, "rule_accuracy.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n已保存 results/rule_accuracy.json")


if __name__ == "__main__":
    main()
