#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相融实验 · 数据采集（供"大胆配色"图表集使用）
=============================================

从已有 checkpoint + 120 条留出轨迹，一次性产出：
  (1) 四类异常 × {方案1(FluxEV), 方案2(max), 方案2+zclip} 的 ROC 曲线点
  (2) S_dl 门控阈值扫描（zclip 标定后）的逐类型 AUROC
  (3) 方案1 / 方案2 / 方案2+zclip 的逐类型 AUROC/AP（与 compare_methods 同口径）

全部落盘到 results/fusion_roc_data.json，供 make_bold_figures.py 出图。
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch
import numpy as np
from sklearn import metrics
import tcn_branch as T

RESULTS = T.RESULTS_DIR
KIND = {"Loitering": T.inject_loiter, "Sprinting": T.inject_run,
        "Zigzag": T.inject_zigzag, "Sudden stop": T.inject_stop}

GATES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
KZ = 3.0   # z-clip 系数


def tcn_frame_ade(model, x, y, obs_len, pred_len):
    n = len(x)
    total = obs_len + pred_len
    ade = np.full(n, np.nan)
    if n < total:
        return ade
    for s in range(0, n - total + 1, 1):
        ref = np.array([x[s + obs_len - 1], y[s + obs_len - 1]])
        obs = np.stack([x[s:s + obs_len], y[s:s + obs_len]], axis=-1) - ref
        fut = np.stack([x[s + obs_len:s + total], y[s + obs_len:s + total]], axis=-1) - ref
        ade[s + obs_len] = float(T.compute_ade(model, obs[None], fut[None])[0])
    return ade


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, 120, seed=7)
    print(f"评估 {len(trajs)} 条轨迹 x 4 类异常 ...")

    roc = {}          # roc[type][method] = {"fpr":[...], "tpr":[...], "auc":float}
    sweep = {}        # sweep[type] = {gate: auc}
    summary = {}      # summary[type][method] = {"auroc":.., "ap":..}

    for en, inject_fn in KIND.items():
        # 累积（测试段逐帧）score / label，三种方法分开
        acc = {"fluxev": ([], []), "max": ([], []), "zclip": ([], [])}
        gate_auc = {g: [] for g in GATES}
        for t in trajs:
            n = len(t[0])
            if n < obs + pred + 30:
                continue
            x, y, lat, lon, tg, _ = t
            i0 = int(n * 0.55)
            i1 = min(i0 + max(16, int(n * 0.18)), n - obs - pred - 2)
            xa, ya = x.copy(), y.copy()
            inject_fn(xa, ya, i0, i1)
            gt = np.zeros(n); gt[i0:i1] = 1

            ade = tcn_frame_ade(model, xa, ya, obs, pred)
            s_dl_sig = 1.0 / (1.0 + np.exp(-((ade - mu) / (sigma + 1e-8) - 2.0)))
            s_dl_sig = np.nan_to_num(s_dl_sig, nan=0.0)
            z = np.clip((np.nan_to_num(ade, nan=mu) - mu) / (KZ * sigma + 1e-8), 0.0, 1.0)

            lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
            res = T.run_pipeline(lat_a, lon_a, tg, 2.0, others=None, zone=None, poly=None, q=0.005)
            s_ev = res["final"]

            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            sev = s_ev[mask]
            s_max = np.maximum(sev, s_dl_sig[mask])
            s_zc = np.maximum(sev, z[mask])

            acc["fluxev"][0].append(sev); acc["fluxev"][1].append(g)
            acc["max"][0].append(s_max); acc["max"][1].append(g)
            acc["zclip"][0].append(s_zc); acc["zclip"][1].append(g)

            for g0 in GATES:
                f = np.maximum(sev, np.where(z[mask] >= g0, z[mask], 0.0))
                gate_auc[g0].append(metrics.roc_auc_score(g, f))

        roc[en] = {}
        summary[en] = {}
        for meth in ("fluxev", "max", "zclip"):
            s = np.concatenate(acc[meth][0]); l = np.concatenate(acc[meth][1])
            fpr, tpr, _ = metrics.roc_curve(l, s)
            auc = metrics.roc_auc_score(l, s)
            # 降采样到 ~120 点，减少 JSON 体积
            if len(fpr) > 120:
                idx = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
                fpr, tpr = fpr[idx], tpr[idx]
            roc[en][meth] = {"fpr": [round(float(v), 4) for v in fpr],
                             "tpr": [round(float(v), 4) for v in tpr],
                             "auc": round(float(auc), 3)}
            summary[en][meth] = {"auroc": round(float(auc), 3),
                                 "ap": round(float(metrics.average_precision_score(l, s)), 3)}
        sweep[en] = {f"{g:.2f}": round(float(np.mean(gate_auc[g])), 3) for g in GATES}
        print(f"  {en:<12} fluxev={summary[en]['fluxev']['auroc']}  "
              f"max={summary[en]['max']['auroc']}  zclip={summary[en]['zclip']['auroc']}")

    out = {"roc": roc, "sweep": sweep, "summary": summary,
           "gates": GATES, "kz": KZ}
    json.dump(out, open(os.path.join(RESULTS, "fusion_roc_data.json"), "w"),
              ensure_ascii=False, indent=2)
    print("完成。fusion_roc_data.json 已输出到 results/")


if __name__ == "__main__":
    main()
