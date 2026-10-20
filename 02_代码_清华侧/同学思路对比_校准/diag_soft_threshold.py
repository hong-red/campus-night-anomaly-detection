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
"""诊断本地 TCN 软分支：z-clip(k=3) 的 0.5 门槛是否太高，最优阈值下 acc 能到多少。"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = _PKGP.CODE_TSINGHUA
for _p in (_PROJ, os.path.join(_PROJ, "tcn_dl_branch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from sklearn import metrics
import geolife_pipeline as gp
import tcn_branch as T

DATA = _PKGP.GEOLIFE_DATA
DT = 2.0


def local_xy(lat, lon):
    lat0, lon0 = lat[0], lon[0]
    return (lon - lon0) * 111320.0 * np.cos(np.radians(lat0)), (lat - lat0) * 110540.0


def xy_to_latlon(x, y, lat0, lon0):
    return lat0 + y / 110540.0, lon0 + x / (111320.0 * np.cos(np.radians(lat0)))


def inject_loiter(x, y, i0, i1, amp=8.0, cycle=16):
    cx, cy = float(np.mean(x[i0:i1])), float(np.mean(y[i0:i1]))
    dx, dy = x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]
    norm = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / norm, dy / norm
    idx = np.arange(i0, i1)
    x[i0:i1] = cx + ux * amp * np.sin(2 * np.pi * (idx - i0) / cycle)
    y[i0:i1] = cy + uy * amp * np.sin(2 * np.pi * (idx - i0) / cycle)


def inject_run(x, y, i0, i1, factor=2.5):
    dx, dy = x - x[i0], y - y[i0]
    x[i0:i1] = x[i0] + dx[i0:i1] * factor
    y[i0:i1] = y[i0] + dy[i0:i1] * factor


def inject_zigzag(x, y, i0, i1, amp=5.0, cycle=20):
    dx, dy = x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]
    norm = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    idx = np.arange(i0, i1)
    x[i0:i1] += px * amp * np.sin(2 * np.pi * (idx - i0) / cycle)
    y[i0:i1] += py * amp * np.sin(2 * np.pi * (idx - i0) / cycle)


def frame_ade(model, x, y, obs_len, pred_len):
    n = len(x)
    total = obs_len + pred_len
    if n < total:
        return np.full(n, np.nan)
    obs_list, fut_list = [], []
    for s in range(0, n - total + 1):
        ref = np.array([x[s + obs_len - 1], y[s + obs_len - 1]])
        obs_list.append(np.stack([x[s:s + obs_len], y[s:s + obs_len]], -1) - ref)
        fut_list.append(np.stack([x[s + obs_len:s + total], y[s + obs_len:s + total]], -1) - ref)
    ade_all = T.compute_ade(model, np.asarray(obs_list, np.float32), np.asarray(fut_list, np.float32))
    ade = np.full(n, np.nan)
    ade[obs_len:obs_len + len(ade_all)] = ade_all
    return ade


def max_stationary_run(speed, dt):
    run = best = 0.0
    for v in speed:
        run = run + dt if v < gp.V_STATIONARY else 0.0
        best = max(best, run)
    return best


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    print(f"TCN obs={obs} pred={pred} mu={mu:.2f} sigma={sigma:.2f}")

    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    clean = []
    for w in walks:
        lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
        speed0, _, _ = gp.trajectory_features(lat, lon, DT)
        if max_stationary_run(speed0, DT) <= 120:
            clean.append(w)
    print(f"轨迹 = {len(clean)}")

    for key in ("wandering", "sprint", "zigzag"):
        ade_pos, ade_neg = [], []
        g_pool, s_pool = [], []
        for w in clean:
            lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
            n = len(lat)
            if n < int(700 / DT):
                continue
            x, y = local_xy(lat, lon)
            i0 = int(n * 0.55); i1 = min(i0 + max(16, int(n * 0.18)), n - 1)
            xa, ya = x.copy(), y.copy()
            if key == "wandering":
                inject_loiter(xa, ya, i0, i1)
            elif key == "sprint":
                inject_run(xa, ya, i0, i1)
            else:
                inject_zigzag(xa, ya, i0, i1)
            lat2, lon2 = xy_to_latlon(xa, ya, lat[0], lon[0])
            x2, y2 = local_xy(lat2, lon2)
            ade = frame_ade(model, x2, y2, obs, pred)
            gt = np.zeros(n); gt[i0:i1] = 1
            mask = np.arange(n) >= n // 2
            g = gt[mask]; a = ade[mask]
            ade_pos.append(a[g == 1]); ade_neg.append(a[g == 0])
            g_pool.append(g); s_pool.append(np.nan_to_num(a, nan=mu))

        g = np.concatenate(g_pool); s = np.concatenate(s_pool)
        ap = np.concatenate(ade_pos); an = np.concatenate(ade_neg)
        print(f"\n[{key}] 异常 ADE 中位={np.nanmedian(ap):.2f}m  正常 ADE 中位={np.nanmedian(an):.2f}m")
        # z-clip k=3 的 0.5 门槛对应 ADE 阈值
        for k in (3.0, 2.0, 1.5, 1.0):
            thr = mu + 0.5 * k * sigma
            pred = (s >= thr).astype(int)
            acc = float((pred == g).mean())
            tp = int(((pred == 1) & (g == 1)).sum()); fp = int(((pred == 1) & (g == 0)).sum())
            fn = int(((pred == 0) & (g == 1)).sum())
            rec = tp / (tp + fn) if tp + fn else 0.0
            prec = tp / (tp + fp) if tp + fp else 0.0
            print(f"  z-clip k={k}: S_dl≥0.5 ⇔ ADE≥{thr:.1f}m  acc={acc:.3f}  P={prec:.3f} R={rec:.3f}")
        # 最优 F1 阈值（直接扫 ADE）
        prec_c, rec_c, thr_c = metrics.precision_recall_curve(g, s)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)
        kk = int(np.argmax(f1_c))
        best_thr = float(thr_c[kk]) if kk < len(thr_c) else float(s.max())
        pred = (s >= best_thr).astype(int)
        acc = float((pred == g).mean())
        tp = int(((pred == 1) & (g == 1)).sum()); fp = int(((pred == 1) & (g == 0)).sum())
        fn = int(((pred == 0) & (g == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  最优F1阈值 ADE≥{best_thr:.1f}m: acc={acc:.3f}  P={prec:.3f} R={rec:.3f} F1={f1_c[kk]:.3f}  AUROC={metrics.roc_auc_score(g, s):.3f}")


if __name__ == "__main__":
    main()
