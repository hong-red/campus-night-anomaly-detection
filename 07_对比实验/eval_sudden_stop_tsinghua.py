# -*- coding: utf-8 -*-
"""急停三检测器 —— 清华 GeoLife（50 条，帧级）复现 geo_sudden_stop_50.json。

三个检测器（吃同一条注入急停后的轨迹、只取第二半段、逐帧打分）：
  fluxev = FluxEV 完整管线最终分数 res["final"]（EVT 运动 + 规则）
  dl     = TCN 软分支 z-clip 分数  zclip(frame_ade)（NaN 帧填 mu，即分数 0）
  rules  = 仅 if 规则里的 dwell 分数 res["dwell"]（管线默认 lo=120s 口径）

轨迹：T.collect_walking_trajectories(seed=1000) 取 50 条；
      急停注入 T.inject_stop(i0=0.55n, i1=i0+max(16,0.18n))。

输出：07_对比实验/结果_JSON/geo_sudden_stop_50.json
      （fluxev / dl / rules 三个键，含 acc/bal_acc/auroc/tp/fp/fn/tn 等）
"""
import os, sys, json
ROOT = r"F:\双分支实验_完整交付"
sys.path.insert(0, ROOT)
import _paths as PK
for p in (PK.CODE_TSINGHUA, os.path.join(PK.CODE_TSINGHUA, "tcn_dl_branch")):
    sys.path.insert(0, p)
import torch
import numpy as np
from sklearn import metrics
import geolife_pipeline as gp
import tcn_branch as T

DT = 2.0
OBS, PRED = 16, 8
OUT = os.path.join(ROOT, "07_对比实验", "结果_JSON", "geo_sudden_stop_50.json")


def frame_ade(model, x, y, obs_len, pred_len):
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


def zclip(ade, mu, sigma, k=3.0):
    ade = np.nan_to_num(ade, nan=mu)
    return np.clip((ade - mu) / (k * sigma + 1e-8), 0.0, 1.0)


def detector_metrics(g, s):
    g = g.astype(int); s = np.asarray(s, dtype=float)
    pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (g == 1)).sum())
    fp = int(((pred == 1) & (g == 0)).sum())
    fn = int(((pred == 0) & (g == 1)).sum())
    tn = int(((pred == 0) & (g == 0)).sum())
    acc = float((pred == g).mean())
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    bal_acc = float((tpr + tnr) / 2.0)
    auroc = float(metrics.roc_auc_score(g, s)) if np.std(s) > 1e-9 else 0.5
    return dict(acc=round(acc, 4), bal_acc=round(bal_acc, 4), auroc=round(auroc, 4),
                neg_frac=round(float((g == 0).mean()), 4),
                frames=int(len(g)), pos=int(g.sum()), neg=int((1 - g).sum()),
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    mu, sigma = stats["mu"], stats["sigma"]
    print(f"TCN obs={cfg['obs_len']} pred={cfg['pred_len']} mu={mu:.2f} sigma={sigma:.2f}")

    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, DT, 50, seed=1000)
    print(f"急停评估轨迹 = {len(trajs)} 条 (seed=1000)")

    pool = {"fluxev": ([], []), "dl": ([], []), "rules": ([], [])}
    n_ok = 0
    for x, y, lat, lon, tg, _ in trajs:
        n = len(x)
        if n < OBS + PRED + 30:
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + max(16, int(n * 0.18)), n - OBS - PRED - 2)
        xa, ya = x.copy(), y.copy()
        T.inject_stop(xa, ya, i0, i1)
        gt = np.zeros(n); gt[i0:i1] = 1
        lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
        res = gp.run_pipeline(lat_a, lon_a, tg, DT, others=None, zone=None, poly=None, q=0.005)
        fluxev = res["final"]
        rules = res["dwell"]
        dl = zclip(frame_ade(model, xa, ya, OBS, PRED), mu, sigma)
        mask = np.arange(n) >= n // 2
        g = gt[mask]
        if g.sum() == 0 or g.sum() == len(g):
            continue
        n_ok += 1
        pool["fluxev"][0].append(g); pool["fluxev"][1].append(fluxev[mask])
        pool["dl"][0].append(g); pool["dl"][1].append(dl[mask])
        pool["rules"][0].append(g); pool["rules"][1].append(rules[mask])

    out = {}
    for d in ("fluxev", "dl", "rules"):
        g = np.concatenate(pool[d][0]); s = np.concatenate(pool[d][1])
        out[d] = detector_metrics(g, s)
        print(f"{d:<8} AUROC={out[d]['auroc']:.4f} acc={out[d]['acc']:.4f} "
              f"bal_acc={out[d]['bal_acc']:.4f} tp={out[d]['tp']} fp={out[d]['fp']} "
              f"fn={out[d]['fn']} tn={out[d]['tn']}")

    print(f"急停有效轨迹 = {n_ok} 条")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
