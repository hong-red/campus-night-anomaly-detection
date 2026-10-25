# -*- coding: utf-8 -*-
"""七事件 × 五检测器消融表（清华 GeoLife，micro + zclip，与 §一 主表同口径）。

口径（与《结论汇总.md》§一 主表逐格对应）：
  6 行为(逗留/闯入/聚集/徘徊/疾跑/蛇形)
      轨迹集 = scan_walking + max_stationary_run<=120 过滤 + n>=350（= final_3cols 主表硬/软）
      仅if规则 = max(dwell[lo=15s], gather, geofence)   —— = 主表「硬」列（lo=15s）
  急停
      轨迹集 = collect_walking seed=1000（= geo_sudden_stop_50 主表急停）
      仅if规则 = res["dwell"]（管线 lo=120s）            —— = §四「仅if规则 0.787」
      FluxEV   = res["final"]                            —— = 主表「急停硬」列（0.754）
  聚合 = micro（第二半段帧拼接 → 单次 AUROC）
  TCN  = z-clip S_dl = clip((ADE-mu)/(3σ), 0, 1)

五个检测器：
  rules  = 仅 if 规则（6行为 lo=15s / 急停 lo=120s）
  fluxev = FluxEV 完整管线 res["final"]
  dl     = TCN 软分支 z-clip（= 主表「软」）
  naive  = max(fluxev, dl)
  gated  = max(fluxev, dl * (dl >= 0.5))

输出：07_对比实验/结果_JSON/fusion_ablation_unified.json
"""
import os
import sys
import json

ROOT = r"F:\双分支实验_完整交付"
sys.path.insert(0, ROOT)
import _paths as PK
for p in (PK.CODE_TSINGHUA, os.path.join(PK.CODE_TSINGHUA, "tcn_dl_branch")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import numpy as np
from sklearn import metrics
import geolife_pipeline as gp
import tcn_branch as T

DT = 2.0
DATA = PK.GEOLIFE_DATA
DWELL_LO = 15
DWELL_HI = gp.DWELL_MIN
OUT = os.path.join(ROOT, "07_对比实验", "结果_JSON", "fusion_ablation_unified.json")


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
    ph = np.sin(2 * np.pi * (idx - i0) / cycle)
    x[i0:i1] = cx + ux * amp * ph
    y[i0:i1] = cy + uy * amp * ph


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
    osc = amp * np.sin(2 * np.pi * (idx - i0) / cycle)
    x[i0:i1] += px * osc
    y[i0:i1] += py * osc


def inject_dwell(lat, lon, n):
    i0 = int(n * 0.55); i1 = min(i0 + int(360 / DT), n - 2)
    lat = lat.copy(); lon = lon.copy()
    lat[i0:i1] = lat[i0]; lon[i0:i1] = lon[i0]
    gt = np.zeros(n); gt[i0:i1] = 1
    return lat, lon, None, None, gt


def inject_geofence(lat, lon, n):
    c = int(n * 0.7)
    zone = (float(lat[c]), float(lon[c]), 60.0)
    gt = (gp.geofence_score(lat, lon, zone) > 0).astype(int)
    return lat, lon, None, zone, gt


def inject_gather(lat, lon, n):
    g0 = int(n * 0.55); g1 = min(g0 + int(300 / DT), n - 1)
    kx = 111320.0 * np.cos(np.radians(lat[g0])); ky = 110540.0
    others = []
    for dx, dy in ((12.0, 0.0), (0.0, 12.0)):
        o_lat = np.full(n, lat[g0] + 200.0 / ky)
        o_lon = np.full(n, lon[g0] + 200.0 / kx)
        o_lat[g0:g1] = lat[g0:g1] + dy / ky
        o_lon[g0:g1] = lon[g0:g1] + dx / kx
        others.append((o_lat, o_lon))
    gt = np.zeros(n); gt[g0:g1] = 1
    return lat, lon, others, None, gt


def inject_motion(lat, lon, n, kind):
    x, y = local_xy(lat, lon)
    i0 = int(n * 0.55); i1 = min(i0 + max(16, int(n * 0.18)), n - 1)
    if kind == "wandering":
        inject_loiter(x, y, i0, i1)
    elif kind == "sprint":
        inject_run(x, y, i0, i1)
    else:
        inject_zigzag(x, y, i0, i1)
    lat2, lon2 = xy_to_latlon(x, y, lat[0], lon[0])
    gt = np.zeros(n); gt[i0:i1] = 1
    return lat2, lon2, None, None, gt


def max_stationary_run(speed, dt):
    run = best = 0.0
    for v in speed:
        if v < gp.V_STATIONARY:
            run += dt; best = max(best, run)
        else:
            run = 0.0
    return best


def hard_score(lat, lon, dt, others, zone):
    speed, _, _ = gp.trajectory_features(lat, lon, dt)
    dwell = gp.stationary_episode_score(speed, dt, DWELL_LO, DWELL_HI)
    gather = gp.gather_score(lat, lon, others, dt) if others else np.zeros(len(lat))
    geof = gp.geofence_score(lat, lon, zone) if zone else np.zeros(len(lat))
    return np.maximum.reduce([dwell, gather, geof])


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


def five_scores(lat, lon, tg, dt, others, zone, dl, rules_lo15=True):
    """rules: lo=15s(6行为,=主表硬) / lo=120s(急停,=§四)。"""
    res = gp.run_pipeline(lat, lon, tg, dt, others=others, zone=zone, poly=None, q=0.005)
    fluxev = res["final"]
    rules = hard_score(lat, lon, dt, others, zone) if rules_lo15 else res["dwell"]
    naive = np.maximum(fluxev, dl)
    gated = np.maximum(fluxev, dl * (dl >= 0.5))
    return {"rules": rules, "fluxev": fluxev, "dl": dl, "naive": naive, "gated": gated}


def micro_auroc(g, s):
    auroc = (metrics.roc_auc_score(g, s)
             if g.sum() not in (0, len(g)) and np.std(s) > 1e-9 else None)
    return round(auroc, 4) if auroc is not None else None


def main():
    model, stats, cfg = T.load_predictor(T.CKPT_PATH)
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    print(f"TCN obs={obs} pred={pred} mu={mu:.2f} sigma={sigma:.2f}")

    out = {}

    # ---- 6 行为：scan + clean（与 §一 final_3cols 同源） ----
    walks = gp.scan_walking_trajectories(DATA, max_files=2000, min_dur=900)
    clean = []
    for w in walks:
        lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
        speed0, _, _ = gp.trajectory_features(lat, lon, DT)
        if max_stationary_run(speed0, DT) <= 120:
            clean.append(w)
    print(f"clean(停留<=120s) = {len(clean)} 条")

    BEHAVIORS = [("geofence", "Intrusion", "闯入"), ("gather", "Gathering", "聚集"),
                 ("dwell", "Lingering", "逗留"), ("wandering", "Wandering", "徘徊"),
                 ("sprint", "Sprint", "疾跑"), ("zigzag", "Zigzag", "蛇形")]
    for key, en, cn in BEHAVIORS:
        pool = {d: ([], []) for d in ("rules", "fluxev", "dl", "naive", "gated")}
        used = 0
        for w in clean:
            lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
            n = len(lat)
            if n < int(700 / DT):
                continue
            if key == "geofence":
                lat, lon, others, zone, gt = inject_geofence(lat, lon, n)
            elif key == "gather":
                lat, lon, others, zone, gt = inject_gather(lat, lon, n)
            elif key == "dwell":
                lat, lon, others, zone, gt = inject_dwell(lat, lon, n)
            else:
                lat, lon, others, zone, gt = inject_motion(lat, lon, n, key)

            x, y = T.latlon_to_local(lat, lon)
            dl = zclip(frame_ade(model, x, y, obs, pred), mu, sigma)
            sc = five_scores(lat, lon, tg, DT, others, zone, dl, rules_lo15=True)

            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            for d in ("rules", "fluxev", "dl", "naive", "gated"):
                pool[d][0].append(g); pool[d][1].append(sc[d][mask])
            used += 1
        out[key] = {d: micro_auroc(np.concatenate(pool[d][0]), np.concatenate(pool[d][1]))
                    for d in ("rules", "fluxev", "dl", "naive", "gated")}
        print(f"{cn:<4} {en:<10} traj={used:>3}  rules={out[key]['rules']} "
              f"fluxev={out[key]['fluxev']} dl={out[key]['dl']} "
              f"naive={out[key]['naive']} gated={out[key]['gated']}")

    # ---- 急停：collect seed=1000，仅if规则用 lo=120s ----
    OBS, PRED = 16, 8
    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, DT, 50, seed=1000)
    pool = {d: ([], []) for d in ("rules", "fluxev", "dl", "naive", "gated")}
    used = 0
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
        dl = zclip(frame_ade(model, xa, ya, OBS, PRED), mu, sigma)
        sc = five_scores(lat_a, lon_a, tg, DT, None, None, dl, rules_lo15=False)
        mask = np.arange(n) >= n // 2
        g = gt[mask]
        if g.sum() == 0 or g.sum() == len(g):
            continue
        for d in ("rules", "fluxev", "dl", "naive", "gated"):
            pool[d][0].append(g); pool[d][1].append(sc[d][mask])
        used += 1
    out["sudden-stop"] = {d: micro_auroc(np.concatenate(pool[d][0]), np.concatenate(pool[d][1]))
                          for d in ("rules", "fluxev", "dl", "naive", "gated")}
    print(f"急停 Sudden-stop traj={used:>3}  rules={out['sudden-stop']['rules']} "
          f"fluxev={out['sudden-stop']['fluxev']} dl={out['sudden-stop']['dl']} "
          f"naive={out['sudden-stop']['naive']} gated={out['sudden-stop']['gated']}")

    json.dump(dict(_meta=dict(level="frame", aggregation="micro", tcn_score="z-clip", dt=DT,
                              rules_6beh="if规则 lo=15s", rules_sudden_stop="res[dwell] lo=120s"),
                   **out), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
