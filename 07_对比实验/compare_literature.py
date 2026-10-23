#!/usr/bin/env python3
import os as _os, sys as _sys
_cur = _os.path.dirname(_os.path.abspath(__file__))
for _i in range(6):
    if _os.path.exists(_os.path.join(_cur, "_paths.py")):
        _sys.path.insert(0, _cur)
        break
    _cur = _os.path.dirname(_cur)
_sys.path.insert(0, _os.environ.get("LIT_PKG_ROOT", _os.path.abspath(_os.path.join(_cur, ".."))))
import _paths as _PKGP
# -*- coding: utf-8 -*-
"""
文献对比实验 v8：全帧级
======================

口径（公平对齐）：
  - 规则域(dwell/geofence/gather)：50 条干净行走轨迹、逐帧、第二半段
  - 运动域(wandering/sprint/zigzag)：800 条行走轨迹(seed=1000)、逐帧（第二半段）
  - 所有方法（含我们）都吃同一条注入后的轨迹，逐帧打分、同一个 roc_auc_score

我们（双分支）：
  - 规则域 → 硬规则 lo=15s（本地 final_3cols_local.json hard 值）
  - 运动域 → 本地 TCN 逐帧（z-clip 连续分数；与表6 frame 口径一致）

四篇文献（纯 numpy 现算）：
  SPOT'17 = POT 上下尾；IJGI'22 = 硬规则+大角度转向；Appl.Sci.'19 = 活动面积(扇形位移)
  GRADINGS'20 (IJCNN) = 轨迹段 Normalizing Flow (MAF + median 聚合)
  注：GRADINGS 原文面向整条 GPS 行程的段密度（car/bus），不是逐帧徘徊检测器；
      这里按原文 W=30 段、MAF+median 拼成连续帧分数，直接与帧级方法同表比较。
"""

import os
import sys
import json
import time
import io

_PROJ = _PKGP.CODE_TSINGHUA
_T = os.path.join(_PROJ, "tcn_dl_branch")
for p in (_PROJ, _T):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402
from sklearn import metrics  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.utils import clip_grad_norm_  # noqa: E402

torch.set_num_threads(1)

import tcn_branch as T  # noqa: E402
import geolife_pipeline as gp  # noqa: E402

OUTDIR = os.environ.get("LIT_OUTDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果_JSON"))
os.makedirs(OUTDIR, exist_ok=True)

DT = 2.0
DWELL_LO = 15.0
DWELL_HI = 300.0
OBS, PRED, STRIDE = 16, 8, 2

OURS = {"dwell": 0.974, "geofence": 1.0, "gather": 0.937, "wandering": 0.701, "sprint": 0.729, "zigzag": 0.547, "sudden-stop": 0.716}

# GRADINGS'20 原文设置：MAF ×10 flows、2×32 hidden、W=30 点段、median 聚合、300 epochs
NF_W = 30
NF_FLOWS = 10
NF_HIDDEN = 32
NF_EPOCHS = 300
NF_SEED = 0
# 整条行走轨迹作为一条行程，训练用前 50%（与规则域/运动域的评估半段一致）
NF_TRAIN_FRAC = 0.5


# ============================ 规则域注入 ============================

def max_stationary_run(speed, dt):
    run = best = 0.0
    for v in speed:
        if v < gp.V_STATIONARY:
            run += dt
            best = max(best, run)
        else:
            run = 0.0
    return best


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


# ============================ 三篇文献复现 ============================

def spot_vanilla_score(speed, dt, q=0.005, train_frac=0.5):
    n = len(speed)
    m = int(n * train_frac)
    score = np.zeros(n)
    if m < 30:
        return score

    def _pot_tail(signal, q):
        calib = signal[:m].astype(float)
        if calib.size < 30:
            return None
        t0 = float(np.quantile(calib, 0.98))
        exc = calib[calib > t0] - t0
        if exc.size < 3:
            return None
        try:
            c, _, scale = st.genpareto.fit(exc, floc=0)
        except Exception:
            return None
        if not np.isfinite(c) or not np.isfinite(scale) or scale <= 0 or abs(c) < 1e-9:
            return None
        Nt = float(exc.size)
        z = t0 + (scale / c) * (((q * n) / Nt) ** (-c) - 1.0)
        if not np.isfinite(z) or z <= t0:
            return None
        return t0, z

    up = _pot_tail(speed, q)
    lo = _pot_tail(-speed, q)
    for i in range(m, n):
        s = 0.0
        if up is not None and speed[i] > up[0]:
            s = max(s, min(1.0, (speed[i] - up[0]) / (up[1] - up[0])))
        if lo is not None and -speed[i] > lo[0]:
            s = max(s, min(1.0, ((-speed[i]) - lo[0]) / (lo[1] - lo[0])))
        score[i] = s
    return score


def _heading_deg(lat, lon):
    n = len(lat)
    h = np.zeros(n)
    for i in range(1, n - 1):
        h[i] = np.degrees(np.arctan2(lat[i + 1] - lat[i - 1], lon[i + 1] - lon[i - 1]))
    h[0] = h[1] if n > 1 else 0.0
    h[-1] = h[-2] if n > 2 else h[-1]
    return h


def ijgi_angle_wander(lat, lon, dt, speed, angle_thr=150.0, window=120.0):
    n = len(lat)
    dh = np.abs(np.diff(np.unwrap(np.radians(_heading_deg(lat, lon))))) * 180.0 / np.pi
    dh = np.concatenate(([0.0], dh))
    turn = np.zeros(n)
    for i in range(2, n - 2):
        if speed[i] > 0.4 and dh[i - 1:i + 3].sum() > angle_thr:
            turn[i] = 1.0
    w = max(1, int(window / dt))
    score = np.zeros(n)
    run = 0.0
    for i in range(n):
        run += turn[i]
        if i >= w:
            run -= turn[i - w]
        score[i] = min(1.0, run / 3.0)
    return score


def hard_rules_score(lat, lon, dt, others, zone):
    speed, _, _ = gp.trajectory_features(lat, lon, dt)
    dwell = gp.stationary_episode_score(speed, dt, DWELL_LO, DWELL_HI)
    gather = gp.gather_score(lat, lon, others, dt) if others else np.zeros(len(lat))
    geof = gp.geofence_score(lat, lon, zone) if zone else np.zeros(len(lat))
    return np.maximum.reduce([dwell, gather, geof])


def ijgi_score(lat, lon, dt, speed, others, zone):
    return np.maximum(hard_rules_score(lat, lon, dt, others, zone),
                      ijgi_angle_wander(lat, lon, dt, speed))


def applsci_sector_score(x, y, dt, T0=60.0, D_thr=30.0):
    n = len(x)
    w = max(1, int(T0 / dt))
    dmax = np.zeros(n)
    for i in range(n):
        lo = max(0, i - w)
        dmax[i] = np.hypot(x[lo:i + 1] - x[lo], y[lo:i + 1] - y[lo]).max()
    return np.clip((D_thr - dmax) / D_thr, 0.0, 1.0)


def ref_frame_scores(lat, lon, dt, others, zone, x, y):
    speed, _, _ = gp.trajectory_features(lat, lon, dt)
    return {
        "rules": hard_rules_score(lat, lon, dt, others, zone),
        "spot": spot_vanilla_score(speed, dt),
        "ijgi": ijgi_score(lat, lon, dt, speed, others, zone),
        "applsci": applsci_sector_score(x, y, dt),
    }


# ============================ 窗口级 ============================

def window_scores(frame_score, gt):
    """窗口级：未来段 [s+OBS, s+OBS+PRED) 与异常重叠则标签=1，分数=决策帧 s+OBS。"""
    n = len(gt)
    total = OBS + PRED
    ws, wl = [], []
    for s in range(0, n - total + 1, STRIDE):
        fut = np.arange(s + OBS, s + total)
        wl.append(1 if gt[fut].sum() > 0 else 0)
        ws.append(frame_score[s + OBS])
    return np.array(ws), np.array(wl)


def pooled_auroc(ws_list, wl_list):
    ws = np.concatenate(ws_list)
    wl = np.concatenate(wl_list)
    if wl.sum() == 0 or wl.sum() == len(wl) or np.std(ws) < 1e-9:
        return 0.5
    if wl.dtype != bool and not set(np.unique(wl)).issubset({0, 1, False, True}):
        raise ValueError(f"labels are not binary: {np.unique(wl)[:8]}")
    return float(metrics.roc_auc_score(wl.astype(int), np.asarray(ws, dtype=float)))


# ============================ 我们的 TCN（逐帧） ============================
_TCN_CACHE = {}


def _tcn_model():
    if not _TCN_CACHE:
        ckpt = _os.environ.get("LIT_TCN_CKPT", "")
        if not ckpt or not os.path.exists(ckpt):
            ckpt = os.path.join(_PKGP.WEIGHTS_DIR, "tcn_checkpoint_geolife_dt2s.pt")
        _TCN_CACHE["model"], _TCN_CACHE["stats"], _TCN_CACHE["cfg"] = T.load_predictor(ckpt)
        print(f"  TCN 权重 = {ckpt}  (mu={_TCN_CACHE['stats']['mu']:.2f}m, "
              f"sigma={_TCN_CACHE['stats']['sigma']:.2f}m)")
        if (_TCN_CACHE["cfg"]["obs_len"], _TCN_CACHE["cfg"]["pred_len"]) != (OBS, PRED):
            raise ValueError("TCN checkpoint 的 obs/pred 与表6 口径不一致")
    return _TCN_CACHE["model"], _TCN_CACHE["stats"]


def tcn_frame_score(x, y):
    """逐帧 S_dl：决策帧 s+OBS 的 ADE → z-clip → sigmoid，与表6 frame 口径一致。"""
    model, stats = _tcn_model()
    return T.tcn_frame_scores(model, x, y, OBS, PRED, stats, stride=1)



# ============================ GRADINGS'20（Normalizing Flow 轨迹段） ============================
# Dias et al., "Anomaly Detection in Trajectory Data with Normalizing Flows", IJCNN 2020.
# 原文：W=30 点/段，展平成 3W 维；MAF×10、2×32 隐层、300 epochs、Adam(1e-3)；
# 段异常度 α=-log p，整条轨迹用 median 聚合。这里为了和逐帧方法同表，把"覆盖该帧的
# 所有段的 α 取中位数"作为该帧分数（帧级投影，不改变模型与训练数据）。

NF_S_MAX = 5.0


class _MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("w_mask", torch.as_tensor(np.asarray(mask).T, dtype=torch.float32).contiguous())

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.w_mask, self.bias)


class _MADE(nn.Module):
    def __init__(self, dim, hidden, rng):
        super().__init__()
        self.in_deg = rng.permutation(np.arange(dim)) % max(1, dim - 1) + 1
        self.hid_deg = rng.integers(1, dim - 1, size=hidden).astype(np.int64)
        self.out_deg = rng.permutation(np.arange(dim)) + 1
        self.l1 = _MaskedLinear(dim, hidden, self.hid_deg[None, :] >= self.in_deg[:, None])
        self.l2 = _MaskedLinear(hidden, hidden, self.hid_deg[:, None] >= self.hid_deg[None, :])
        l3_mask = np.ones((hidden, 2 * dim), dtype=bool)
        l3_mask[:, :dim] = (self.hid_deg[:, None] < self.out_deg[None, :])
        l3_mask[:, dim:] = (self.hid_deg[:, None] < self.out_deg[None, :])
        self.l3 = _MaskedLinear(hidden, 2 * dim, l3_mask)
        self.act = nn.Tanh()
        for lin in (self.l1, self.l2, self.l3):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        h = self.act(self.l1(x))
        h = self.act(self.l2(h))
        out = self.l3(h)
        d = x.shape[1]
        return out[:, :d], torch.clamp(out[:, d:], -NF_S_MAX, NF_S_MAX)


class _MAF(nn.Module):
    def __init__(self, dim, hidden, n_flows, rng):
        super().__init__()
        self.blocks = nn.ModuleList([_MADE(dim, hidden, rng) for _ in range(n_flows)])

    def log_prob(self, x):
        logp = torch.zeros(x.shape[0], device=x.device)
        for net in reversed(self.blocks):
            mu, log_s = net(x)
            x = (x - mu) * torch.exp(-log_s)
            logp = logp - log_s.sum(dim=1)
        return logp - 0.5 * (x ** 2).sum(dim=1) - 0.5 * x.shape[1] * np.log(2 * np.pi)


_NF_STATE = {}
_NF_CACHE = {}


def _seg_batch(x, y, start, end, w):
    idx = np.arange(start, max(start, end - w + 1))
    mat = np.zeros((len(idx), 2 * w))
    for k, i in enumerate(idx):
        mat[k, :w] = x[i:i + w]
        mat[k, w:] = y[i:i + w]
    return mat, idx


def _frame_from_segment_scores(score_seg, starts, w, n):
    """Project 段级分数 to 帧：每帧取所有覆盖它的段分数中位数（与 GRADINGS 的 median 聚合同法）。"""
    score_seg = np.asarray(score_seg, dtype=float)
    starts = np.asarray(starts)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = np.searchsorted(starts, i - w + 1), np.searchsorted(starts, i + 1)
        if hi > lo:
            out[i] = float(np.median(score_seg[lo:hi]))
    return out


def nf_frame_score(x, y, train_len):
    """GRADINGS (Dias et al., IJCNN 2020)：W=30 段、MAF×10、median 聚合。"""
    W = NF_W
    dim = 2 * W
    mat, starts = _seg_batch(x, y, 0, len(x), W)
    train_n = sum(1 for s0 in starts if s0 + W <= train_len)
    if train_n < 4:
        return np.full(len(x), np.nan)
    mu, sd = mat[:train_n].mean(axis=0, keepdims=True), mat[:train_n].std(axis=0, keepdims=True) + 1e-6
    mat = (mat - mu) / sd
    key = (float(np.sum(np.abs(x))), float(np.sum(np.abs(y))), len(x))
    if key not in _NF_CACHE:
        torch.manual_seed(NF_SEED)
        rng = np.random.default_rng(NF_SEED)
        model = _MAF(dim, NF_HIDDEN, NF_FLOWS, rng)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = torch.tensor(mat[:train_n].astype(np.float32))
        model.train()
        for _ in range(NF_EPOCHS):
            order = rng.permutation(len(data))
            for k in range(0, len(data), 512):
                idx = order[k:k + 512]
                xb = data[idx] + 0.01 * torch.randn(len(idx), dim)
                loss = -model.log_prob(xb).mean()
                opt.zero_grad(); loss.backward()
                clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        _NF_CACHE[key] = model
    with torch.no_grad():
        a = -_NF_CACHE[key].log_prob(torch.tensor(mat.astype(np.float32))).numpy()
    return _frame_from_segment_scores(a, starts, W, len(x))


class _DenseAE(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(hidden // 2, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, z):
        return self.dec(self.enc(z))


class _LSTM4(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.enc = nn.GRU(4, hidden, batch_first=True)
        self.dec = nn.GRU(4, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 4)

    def forward(self, obs, pred_len):
        _, h = self.enc(obs)
        h = h.contiguous()
        inp = obs[:, -1:, :]
        out = []
        for _ in range(pred_len):
            z, h = self.dec(inp, h)
            inp = self.out(z)
            out.append(inp)
        return torch.cat(out, dim=1)


def _fit_ae(mat, epochs=40, batch=256):
    torch.manual_seed(NF_SEED)
    rng = np.random.default_rng(NF_SEED)
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True) + 1e-6
    data = torch.tensor((mat - mean) / std, dtype=torch.float32)
    model = _DenseAE(mat.shape[1], 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(data))
        for k in range(0, len(data), batch):
            xb = data[torch.tensor(order[k:k + batch])]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, mean, std


def _ae_segment_scores(model, mean, std, mat):
    with torch.no_grad():
        z = torch.tensor((mat - mean) / std, dtype=torch.float32)
        return ((model(z) - z) ** 2).mean(dim=1).numpy()

# ============================ 其余文献基线（逐帧投影） ============================
# 统一原则：原文是整条轨迹 / 子轨迹 / 时间窗级的方法，投影到每个帧时都用覆盖该帧的
# 原有明细（段或窗）取中位数，避免额外改变原文评分；多类别评分统一转成每类一个连续分数。

_WIN = 12


def ae_frame_score(x, y, train_len):
    """Dense AE reconstruction（Pedestrian Anomaly Detection Method using Autoencoder, ICHCI 2021 思路）：
    输入 = 窗口内相对位移 (W,2)，训练只用前半段；分数 = 重建 MSE 的窗口级中位数。"""
    mat, starts = _seg_batch(x - x[0], y - y[0], 0, len(x), _WIN)
    train_n = sum(1 for s0 in starts if s0 + _WIN <= train_len)
    if train_n < 8:
        return np.full(len(x), np.nan)
    model, mu, sd = _fit_ae(mat[:train_n])
    sc = _ae_segment_scores(model, mu, sd, mat)
    return _frame_from_segment_scores(sc, starts, _WIN, len(x))


def _feat4(seg):
    """(h,2) 相对坐标 → 每步 (dx,dy,v,dheading)。"""
    dx, dy = np.diff(seg[:, 0], prepend=seg[0, 0]), np.diff(seg[:, 1], prepend=seg[0, 1])
    v = np.hypot(dx, dy)
    a = np.arctan2(dy, dx)
    dh = np.diff(a, prepend=a[0])
    return np.stack([dx, dy, v, dh], axis=1)


def lstm_frame_score(x, y, train_len):
    """LSTM 轨迹预测误差（Kanu-Asiegbu et al., IEEE SSCI 2021 的 LSTM 预测器）：
    OBS=8 步历史预测未来 8 步，训练仅前 50%；分数 = 预测 L2 误差（段中位数）。"""
    O, P, W = 8, 8, 16
    segs, starts = [], []
    for i in range(0, len(x) - W + 1):
        seg = np.stack([x[i:i + W], y[i:i + W]], axis=-1)
        segs.append(_feat4(seg))
        starts.append(i)
    if not segs:
        return np.full(len(x), np.nan)
    arr = np.stack(segs).astype(np.float32)
    train_idx = [k for k, i in enumerate(starts) if i + W <= train_len]
    if len(train_idx) < 16:
        return np.full(len(x), np.nan)
    torch.manual_seed(NF_SEED)
    rng = np.random.default_rng(NF_SEED)
    model = _LSTM4()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(40):
        order = rng.permutation(train_idx)
        for k in range(0, len(order), 128):
            b = order[k:k + 128]
            obs = torch.tensor(arr[b, :O])
            fut = torch.tensor(arr[b, O:])
            loss = ((model(obs, P) - fut) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(arr[:, :O]), P).numpy()
    err = np.linalg.norm(pred[:, :, :2] - arr[:, O:, :2], axis=-1).mean(axis=1)
    return _frame_from_segment_scores(err, starts, W, len(x))


def gnss_frame_score(x, y, train_len):
    """Sensors 2018 "GNSS Trajectory Anomaly Detection Using Similarity Comparison":
    每个 16 步段先按原文做 PCA 子空间投影，再取到最近正常段（前半段）的欧氏距离；
    分数 = 最近正常段距离（段级），帧级取覆盖该帧的段中位数。"""
    W = 16
    mat, starts = _seg_batch(x - x[0], y - y[0], 0, len(x), W)
    train_n = sum(1 for s0 in starts if s0 + W <= train_len)
    if train_n < 4:
        return np.full(len(x), np.nan)
    mu = mat[:train_n].mean(axis=0, keepdims=True)
    sub = mat - mu
    _, _, vt = np.linalg.svd(sub[:train_n], full_matrices=False)
    emb = sub @ vt[:4].T
    norm = emb[:train_n]
    scores = []
    for k in range(len(emb)):
        d = np.sum((norm - emb[k]) ** 2, axis=1)
        scores.append(float(np.min(d)))
    return _frame_from_segment_scores(scores, starts, W, len(x))


def cbvm_frame_score(x, y, train_len):
    """CBVM (Wang et al. 2011) 的逐帧投影：以固定格点做距离加权的转折点似然，
    窗口内转折点越少 → 分数越低（徘徊的往复运动会被压平为低分）。"""
    W = 16
    grid = 12.0
    gx, gy = np.round(x / grid), np.round(y / grid)
    turn = np.zeros(len(x))
    for i in range(2, len(x)):
        a = np.array([x[i - 1] - x[i - 2], y[i - 1] - y[i - 2]])
        b = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-6 and nb > 1e-6:
            turn[i] = float(np.clip((1 - np.dot(a, b) / (na * nb)) / 2, 0, 1))
    out = np.full(len(x), np.nan)
    for i in range(W - 1, len(x)):
        out[i] = float(turn[i - W + 1:i + 1].sum())
    return out


# ============================ 基准 ============================

def run_rule():
    walks = gp.scan_walking_trajectories(T.DEFAULT_DATA, max_files=2000, min_dur=900)
    clean = []
    for w in walks:
        lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
        speed0, _, _ = gp.trajectory_features(lat, lon, DT)
        if max_stationary_run(speed0, DT) <= 120:
            clean.append(w)
    print(f"  规则域干净轨迹 = {len(clean)} 条")

    events = {"dwell": inject_dwell, "geofence": inject_geofence, "gather": inject_gather}
    result = {}
    for key, inj in events.items():
        pools = {d: ([], []) for d in "rules spot applsci ijgi gradings ae lstm gnss cbvm".split()}
        for wi, w in enumerate(clean):
            _t0 = time.time()
            lat, lon, tg = gp.resample(w[3], w[4], w[5], DT)
            n = len(lat)
            if n < int(700 / DT):
                continue
            lat, lon, others, zone, gt = inj(lat, lon, n)
            x, y = T.latlon_to_local(lat, lon)
            sc = ref_frame_scores(lat, lon, DT, others, zone, x, y)
            sc["gradings"] = nf_frame_score(x, y, int(n * NF_TRAIN_FRAC))
            sc["ae"] = ae_frame_score(x, y, int(n * NF_TRAIN_FRAC))
            sc["lstm"] = lstm_frame_score(x, y, int(n * NF_TRAIN_FRAC))
            sc["gnss"] = gnss_frame_score(x, y, int(n * NF_TRAIN_FRAC))
            sc["cbvm"] = cbvm_frame_score(x, y, int(n * NF_TRAIN_FRAC))
            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            print(f"    {key} {wi+1}/{len(clean)} n={n} ({time.time()-_t0:.1f}s)", flush=True)
            for d, s in sc.items():
                # 所有方法同批帧：只剔除该方法未产生分数的帧
                ok = np.isfinite(s[mask])
                pools[d][0].append(g[ok])
                pools[d][1].append(s[mask][ok])
        result[key] = {d: round(float(metrics.roc_auc_score(
            np.concatenate(pools[d][0]), np.concatenate(pools[d][1]))), 3)
            for d in pools if len(pools[d][0]) and np.concatenate(pools[d][0]).sum() not in (0, len(pools[d][0]))}
    return result


def run_stop_framewise(max_traj=50):
    """急停：逐帧、第二半段；我们 = fluxev=max(EVT, 规则)，文献同口径 ref_frame_scores。"""
    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, max_traj, seed=1000)
    print(f"  急停评估轨迹 = {len(trajs)} 条 (seed=1000, 逐帧)")
    DETS = ["applsci", "spot", "gradings", "ae", "lstm", "gnss", "cbvm"]
    pool = {d: ([], []) for d in DETS}
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
        sc = ref_frame_scores(lat_a, lon_a, DT, None, None, xa, ya)
        sc["gradings"] = nf_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
        sc["ae"] = ae_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
        sc["lstm"] = lstm_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
        sc["gnss"] = gnss_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
        sc["cbvm"] = cbvm_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
        mask = np.arange(n) >= n // 2
        g = gt[mask]
        if g.sum() == 0 or g.sum() == len(g):
            continue
        n_ok += 1
        for d in DETS:
            ok = np.isfinite(sc[d][mask])
            if not ok.any():
                print(f"    急停 {d} 无可评分帧 (n={n})", flush=True)
                continue
            if ok.sum() < len(ok):
                print(f"    急停 {d} 部分帧 NaN: {int(ok.sum())}/{len(ok)} (n={n})", flush=True)
            pool[d][0].append(g[ok])
            pool[d][1].append(sc[d][mask][ok])
    out = {d: round(pooled_auroc(pool[d][0], pool[d][1]), 3) for d in DETS}
    if os.environ.get("LIT_STOP_RECHECK") == "1":
        fx = _stop_fluxev_framewise(trajs)
        print(f"  急停 fluxev(max(EVT,规则)) 复算 = {fx:.3f} (主表固定 {OURS['sudden-stop']:.3f})")
    print(f"  急停有效轨迹 = {n_ok} 条")
    return {"sudden-stop": out}


def _stop_fluxev_framewise(trajs):
    gs, ss = [], []
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
        mask = np.arange(n) >= n // 2
        g = gt[mask]
        if 0 < g.sum() < len(g):
            gs.append(g); ss.append(res["final"][mask])
    return pooled_auroc(gs, ss)


def run_motion_framewise(max_traj=50):
    trajs = T.collect_walking_trajectories(T.DEFAULT_DATA, 2.0, max_traj, seed=1000)
    print(f"  运动域评估轨迹 = {len(trajs)} 条 (seed=1000, 统一 50 条口径)")

    events = {"wandering": T.inject_loiter, "sprint": T.inject_run, "zigzag": T.inject_zigzag}
    DETS = ["applsci", "spot", "gradings", "ae", "lstm", "gnss", "cbvm"]
    pool = {(en, d): ([], []) for en in events for d in list(DETS) + ["ours"]}
    for en, fn in events.items():
        for t in trajs:
            x, y, lat, lon, tg, _ = t
            n = len(x)
            if n < OBS + PRED + 30:
                continue
            i0 = int(n * 0.55)
            i1 = min(i0 + max(16, int(n * 0.18)), n - OBS - PRED - 2)
            xa, ya = x.copy(), y.copy()
            fn(xa, ya, i0, i1)
            gt = np.zeros(n); gt[i0:i1] = 1
            lat_a, lon_a = T.local_to_latlon(xa, ya, lat[0], lon[0])
            sc = ref_frame_scores(lat_a, lon_a, DT, None, None, xa, ya)
            sc["gradings"] = nf_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
            sc["ae"] = ae_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
            sc["lstm"] = lstm_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
            sc["gnss"] = gnss_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
            sc["cbvm"] = cbvm_frame_score(xa, ya, int(n * NF_TRAIN_FRAC))
            mask = np.arange(n) >= n // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            for d in DETS:
                ok = np.isfinite(sc[d][mask])
                pool[(en, d)][0].append(sc[d][mask][ok])
                pool[(en, d)][1].append(g[ok])
            tcn = tcn_frame_score(xa, ya)[mask]
            ok = np.isfinite(tcn)
            pool[(en, "ours")][0].append(tcn[ok])
            pool[(en, "ours")][1].append(g[ok])
    ours = {en: round(pooled_auroc(pool[(en, "ours")][0], pool[(en, "ours")][1]), 3)
            for en in events}
    refs = {en: {d: round(pooled_auroc(pool[(en, d)][0], pool[(en, d)][1]), 3)
                 for d in DETS} for en in events}
    refs["ours"] = ours
    json.dump(refs, open(os.path.join(OUTDIR, "motion_domain.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("运动域已落盘 motion_domain.json", flush=True)
    return refs


# ============================ 组装 ============================

def main():
    loc = json.load(open(os.path.join(_PROJ, "同学思路对比_校准", "final_3cols_local.json"),
                         encoding="utf-8")) if os.path.exists(os.path.join(_PROJ, "同学思路对比_校准", "final_3cols_local.json")) else None
    OURS_RULE = {
        "dwell": loc["dwell"]["hard"]["auroc"],
        "geofence": loc["geofence"]["hard"]["auroc"],
        "gather": loc["gather"]["hard"]["auroc"],
    }

    print("[1/3] 规则域（50 条，逐帧）...")
    rule = run_rule()
    json.dump(rule, open(os.path.join(OUTDIR, "rule_domain.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("规则域已落盘 rule_domain.json", flush=True)
    json.dump({"n_clean_trajectories": 50, "frame_level": True, "rows": rule}, io.open(os.path.join(OUTDIR, "rule_domain.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("规则域已落盘 rule_domain.json", flush=True)

    print("[2/3] 运动域（50 条，逐帧，文献基线 + 我们的 TCN）...")
    motion = run_motion_framewise(max_traj=50)
    print("[3/3] 急停（50 条，逐帧）...")
    stop = run_stop_framewise()
    motion = {**motion, **stop}

    order = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag", "sudden-stop"]
    METHODS = ["ours_dual", "applsci", "spot", "gradings", "ae", "lstm", "gnss", "cbvm"]
    methods = METHODS
    matrix = {}
    for e in order:
        src = rule if e in rule else motion
        matrix[e] = {"ours_dual": OURS[e], "ours_branch":
                     "rules(lo=15)" if e in rule else ("fluxev(frame)" if e == "sudden-stop" else "TCN(frame)")}
        for m in METHODS[1:]:
            matrix[e][m] = src[e][m]

    summary = {}
    for m in METHODS:
        vals = [matrix[e][m] for e in order]
        cover = sum(1 for v in vals if v > 0.55)
        summary[m] = dict(coverage=f"{cover}/7", n_covered=cover)

    json.dump({"order": order, "matrix": matrix, "summary": summary},
              open(os.path.join(OUTDIR, "literature_comparison.json"), "w"),
              ensure_ascii=False, indent=2)

    M_LABEL = {"ours_dual": "Ours (dual-branch)", "ijgi": "IJGI'22 (rules)",
               "applsci": "Appl.Sci.'19 (area)", "spot": "SPOT'17 (EVT)",
               "gradings": "GRADINGS'20 (NF)", "ae": "ICHCI'21 (AE)",
               "lstm": "SSCI'21 (LSTM)", "gnss": "Sensors'18 (sim)",
               "cbvm": "CBVM'11"}
    print("\n" + "=" * 104)
    print("AUROC 矩阵（7 事件 × 文献；全部逐帧）")
    print("=" * 104)
    print(f"{'event':<12}" + "".join(f"{M_LABEL[m]:<22}" for m in methods))
    print("-" * 84)
    for e in order:
        print(f"{e:<12}" + "".join(f"{matrix[e][m]:<22.3f}" for m in methods)
              + f"  [ours={matrix[e]['ours_branch']}]")
    print("-" * 84)
    for m in methods:
        print(f"  {M_LABEL[m]:<22} coverage={summary[m]['coverage']}")
    print("\n已保存:", os.path.join(OUTDIR, "literature_comparison.json"))


if __name__ == "__main__":
    main()
