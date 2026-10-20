#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相融实验 · 补充：S_dl 标定策略对照
==================================

上一轮发现：naive max 在急停上掉到 0.368，而"绝对阈值门控(≥0.5)"虽救回急停
却把运动三类增益全打回 0.5。根因是 ade_to_score 用 sigmoid((ADE-mu)/sigma - 2)
把分数压到 0.12~0.14 的窄带里——信号和噪声都在低值区。

本脚本直接拿"逐帧原始 ADE"，换成不同标定函数后做 max 融合，看哪种标定能：
  ① 保留运动三类(疾跑/徘徊/蛇形)的增益
  ② 让急停回到 FluxEV-only 的 0.618 附近（TCN 对该类本就无信号，别添乱）

标定函数（均映射到 [0,1]，再与 S_ev 取 max）：
  sigmoid(offset)  1/(1+exp(-((ADE-mu)/sigma - offset)))   offset∈{2,1,0}
  zclip(k)         clip((ADE-mu)/(k*sigma), 0, 1)          k∈{2,3,4}
  rank             该条轨迹内 ADE 的百分位排名（相对异常上界）

输出：results/fusion_recalib.json + 打印表。
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
from scipy.stats import rankdata
from sklearn import metrics
import tcn_branch as T

RESULTS = T.RESULTS_DIR
KIND = {"loiter": ("Loitering", T.inject_loiter),
        "sprint": ("Sprinting", T.inject_run),
        "zigzag": ("Zigzag", T.inject_zigzag),
        "stop": ("Sudden stop", T.inject_stop)}


def tcn_frame_ade(model, x, y, obs_len, pred_len):
    """与 tcn_frame_scores 同口径，但返回逐帧原始 ADE(米)，无窗口处为 NaN。"""
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


def calib(name, ade, mu, sigma):
    ade = np.nan_to_num(ade, nan=mu)     # 无窗口处按正常 ADE 处理，不引入异常
    if name.startswith("sigmoid"):
        offset = float(name.split("=")[1])
        z = (ade - mu) / (sigma + 1e-8) - offset
        return 1.0 / (1.0 + np.exp(-z))
    if name.startswith("zclip"):
        k = float(name.split("=")[1])
        return np.clip((ade - mu) / (k * sigma + 1e-8), 0.0, 1.0)
    if name == "rank":
        r = rankdata(ade) / len(ade)      # (0,1]，此轨迹内相对排名
        return r
    raise ValueError(name)


CALIBS = ["sigmoid=2", "sigmoid=1", "sigmoid=0", "zclip=2", "zclip=3", "zclip=4", "rank"]


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, 120, seed=7)
    print(f"评估 {len(trajs)} 条轨迹 x {len(KIND)} 类异常 x {len(CALIBS)} 种标定 ...")

    agg = {(en, c): [] for en, _ in KIND.values() for c in CALIBS}

    for key, (en, fn) in KIND.items():
        for t in trajs:
            n = len(t[0])
            if n < obs + pred + 30:
                continue
            x, y, lat, lon, tg, _ = t
            i0 = int(n * 0.55)
            i1 = min(i0 + max(16, int(n * 0.18)), n - obs - pred - 2)
            xa, ya = x.copy(), y.copy()
            fn(xa, ya, i0, i1)
            gt = np.zeros(n); gt[i0:i1] = 1

            ade = tcn_frame_ade(model, xa, ya, obs, pred)
            lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
            res = T.run_pipeline(lat_a, lon_a, tg, 2.0, others=None, zone=None, poly=None, q=0.005)
            s_ev = res["final"]

            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            for c in CALIBS:
                s_dl = calib(c, ade, mu, sigma)
                fused = np.maximum(s_ev, s_dl)
                agg[(en, c)].append(metrics.roc_auc_score(g, fused[mask]))

    print("\n" + "=" * 84)
    print("S_dl 标定策略 × 四类异常 AUROC（max 融合，逐轨迹均值）")
    print("=" * 84)
    print(f"{'calibration':<14}" + "".join(f"{en:<13}" for en, _ in KIND.values()))
    print("-" * 84)
    out = {}
    for c in CALIBS:
        out[c] = {}
        row = f"{c:<14}"
        for en, _ in KIND.values():
            a = float(np.mean(agg[(en, c)]))
            out[c][en] = round(a, 3)
            row += f"{a:<13.3f}"
        print(row)

    json.dump(out, open(os.path.join(RESULTS, "fusion_recalib.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n完成。fusion_recalib.json 已输出到 results/")


if __name__ == "__main__":
    main()
