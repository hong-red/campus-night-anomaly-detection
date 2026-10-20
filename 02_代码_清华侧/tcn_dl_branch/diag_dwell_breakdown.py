#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：逗留(dwell) 的 acc/P/R/F1 到底被什么吃掉 —— 复现 compute_rule_accuracy 的同一条轨迹，
把 dwell 单独规则 vs 合并规则(max) 拆开，并定位 FP/FN 帧落在哪些区域。"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch
import tcn_branch as T
import geolife_pipeline as gp
import fusion_ablation as FA


def cm(score, g, thr=0.5):
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


def main():
    model, stats, cfg = T.load_predictor(FA.CKPT)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]

    walks = gp.scan_walking_trajectories(T.DEFAULT_DATA, max_files=2000, min_dur=900)
    npts, med, path, lat5, lon5, tg5 = walks[0]
    lat, lon, tg = gp.resample(lat5, lon5, tg5, 2.0)
    lat, lon, t, gt, others, zone = FA.build_rule_scenario(lat, lon, tg, 2.0)
    n = len(lat)
    dt = 2.0

    res = gp.run_pipeline(lat, lon, t, 2.0, others=others, zone=zone, q=0.005)
    dwell_only = res["dwell"]
    rules = np.maximum.reduce([res["dwell"], res["gather"], res["geofence"]])

    mask = np.arange(n) >= n // 2
    g = gt["dwell"][mask]
    test_start = n // 2

    # 注入区间（帧索引，全局）
    d0 = test_start + int(150 / dt)
    d1 = d0 + int(360 / dt)
    g0 = test_start + int(1200 / dt)
    g1 = g0 + int(300 / dt)

    print("=" * 88)
    print(f"轨迹: {os.path.basename(path)}  重采样后 n={n} 帧 (dt={dt}s)  测试段 {mask.sum()} 帧")
    print(f"dwell 注入: 帧[{d0},{d1}) 即 t=[{d0*dt}s, {d1*dt}s)  360s/180帧")
    print(f"gather 注入: 帧[{g0},{g1}) 即 t=[{g0*dt}s, {g1*dt}s)  300s/150帧")
    print(f"geofence: 中心 t≈{ (test_start+int(2100/dt))*dt }s, r=60m")
    print("=" * 88)

    print(f"\n[dwell GT] 测试段正样本 {g.sum()} 帧, 占比 {g.mean()*100:.2f}%  "
          f"(全判负基线 acc={(1-g.mean())*100:.2f}%)")

    r_dwell = cm(dwell_only[mask], g)
    r_comb = cm(rules[mask], g)
    print(f"\n① 只用 dwell 规则:        acc={r_dwell['acc']:.4f}  P={r_dwell['prec']:.4f}  "
          f"R={r_dwell['rec']:.4f}  F1={r_dwell['f1']:.4f}  [TP{r_dwell['tp']} FP{r_dwell['fp']} FN{r_dwell['fn']} TN{r_dwell['tn']}]")
    print(f"② 合并规则 max(d,g,f):    acc={r_comb['acc']:.4f}  P={r_comb['prec']:.4f}  "
          f"R={r_comb['rec']:.4f}  F1={r_comb['f1']:.4f}  [TP{r_comb['tp']} FP{r_comb['fp']} FN{r_comb['fn']} TN{r_comb['tn']}]")

    # dwell 规则的分数随停留时长的爬升（验证 FN=105 是否纯属线性抬升阈值假象）
    # 注入段内 dwell_only 分数序列
    seg = dwell_only[d0:d1]
    print(f"\n③ 注入 dwell 段内 dwell_only 分数: 前10帧 {np.round(seg[:10],3)}  "
          f"... 后10帧 {np.round(seg[-10:],3)}")
    print(f"   分数>=0.5 的帧数 = {(seg>=0.5).sum()} / {len(seg)}")
    # 理论上 0.5 阈值需 run>=210s=105帧, 即从第 105 帧起才算正 -> 后 75 帧
    ramp_frames = int((0.5*(gp.DWELL_MIN-gp.DWELL_LOW)+gp.DWELL_LOW)/dt)
    print(f"   理论: score=(run-{gp.DWELL_LOW})/({gp.DWELL_MIN}-{gp.DWELL_LOW}), "
          f">=0.5 需 run>={ramp_frames*dt}s={ramp_frames}帧 -> 后 {len(seg)-ramp_frames} 帧为正")

    # 定位 FP 帧落在哪个区域（合并规则口径）
    pred_comb = (np.nan_to_num(rules) >= 0.5).astype(int)
    fp_idx = np.where((pred_comb == 1) & (gt["dwell"] == 0) & mask)[0]
    # 归到几个区间
    regions = {
        "dwell注入段前(测试段头部)": (test_start, d0),
        "gather注入段[g0,g1)": (g0, g1),
        "geofence圈内": None,
        "其他(真实慢速/停顿)": None,
    }
    print(f"\n④ 合并规则口径的 {len(fp_idx)} 个 FP 帧分布:")
    bins = {
        f"测试段头部[test_start,d0)={test_start},{d0}": ((fp_idx >= test_start) & (fp_idx < d0)).sum(),
        f"dwell注入段内(应算TP不算FP)": 0,
        f"gather段[g0,g1)={g0},{g1}": ((fp_idx >= g0) & (fp_idx < g1)).sum(),
        f"geofence触发段(距离zone<60m)": 0,
        "其余(真实停顿/慢速)": 0,
    }
    # geofence 触发帧
    gf = (gp.geofence_score(lat, lon, zone) > 0).astype(int)
    gf_idx = np.where((gf == 1) & (gt["dwell"] == 0) & mask)[0]
    in_gf = np.isin(fp_idx, gf_idx).sum()
    bins["geofence触发段(距离zone<60m)"] = in_gf
    # 其余
    assigned = ((fp_idx >= test_start) & (fp_idx < d0)).sum() + \
               ((fp_idx >= g0) & (fp_idx < g1)).sum() + in_gf
    bins["其余(真实停顿/慢速)"] = len(fp_idx) - assigned
    for k, v in bins.items():
        if v:
            print(f"    {v:>5} 帧  <- {k}")

    # dwell_only 口径的 FP 分布（真实 dwell 规则误报）
    pred_d = (np.nan_to_num(dwell_only) >= 0.5).astype(int)
    fp_d = np.where((pred_d == 1) & (gt["dwell"] == 0) & mask)[0]
    print(f"\n⑤ 只用 dwell 规则口径的 {len(fp_d)} 个 FP 帧（这才是 dwell 规则自己的误报）:")
    # 这些 FP 帧的 speed 分布
    speed, _, _ = gp.trajectory_features(lat, lon, dt)
    if len(fp_d):
        print(f"    这些帧 speed 分位 [0,25,50,75,100]% = "
              f"{np.round(np.percentile(speed[fp_d], [0,25,50,75,100]),3)} m/s")
        # 最长连续 FP 段
        dseg = np.diff(fp_d)
        runs = np.split(fp_d, np.where(dseg > 1)[0] + 1)
        longest = max(runs, key=len)
        print(f"    最长连续误报段 = {len(longest)} 帧 = {len(longest)*dt}s "
              f"(帧[{longest[0]},{longest[-1]}))")


if __name__ == "__main__":
    main()
