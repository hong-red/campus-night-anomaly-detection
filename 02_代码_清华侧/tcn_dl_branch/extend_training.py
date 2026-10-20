#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""延长训练到 35 轮（与满规模同 seed/同配置），验证 25 轮是否仍是验证集最优。"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch
import torch.nn.functional as F
import numpy as np

from tcn_branch import (TCNPredictor, collect_walking_trajectories, windowize,
                        DEFAULT_DATA, RESULTS_DIR)

SEED = 1
EPOCHS = 35
MAX_TRAJ = 800
MAX_WINDOWS = 100000
DT = 2.0
OBS, PRED, STRIDE = 16, 8, 2
HIDDEN, LAYERS, K, DROPOUT, LR, BATCH = 32, 4, 3, 0.1, 1e-3, 256

torch.manual_seed(SEED)
np.random.seed(SEED)

print("扫描 Geolife 行走轨迹 ...", flush=True)
trajs = collect_walking_trajectories(DEFAULT_DATA, DT, MAX_TRAJ, SEED)
print(f"收集到 {len(trajs)} 条，切分窗口 ...", flush=True)
obs_arr, fut_arr = windowize(trajs, OBS, PRED, STRIDE, MAX_WINDOWS)
print(f"样本窗口数 = {len(obs_arr)}", flush=True)

idx = np.random.default_rng(SEED).permutation(len(obs_arr))
n_val = int(len(obs_arr) * 0.1)
val_idx, tr_idx = idx[:n_val], idx[n_val:]
print(f"训练 {len(tr_idx)} / 验证 {len(val_idx)}", flush=True)

tr_x = torch.tensor(obs_arr[tr_idx], dtype=torch.float32)
tr_y = torch.tensor(fut_arr[tr_idx], dtype=torch.float32)
va_x = torch.tensor(obs_arr[val_idx], dtype=torch.float32)
va_y = torch.tensor(fut_arr[val_idx], dtype=torch.float32)

model = TCNPredictor(in_dim=2, hidden=HIDDEN, layers=LAYERS, k=K,
                     obs_len=OBS, pred_len=PRED, dropout=DROPOUT)
n_param = sum(p.numel() for p in model.parameters())
print(f"模型参数量 = {n_param}", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=LR)

tr_losses, va_losses = [], []
best_va, best_ep = float("inf"), -1
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(tr_x))
    tot, cnt = 0.0, 0
    for i in range(0, len(tr_x), BATCH):
        b = perm[i:i + BATCH]
        loss = F.mse_loss(model(tr_x[b]), tr_y[b])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss) * len(b); cnt += len(b)
    tr = tot / cnt
    va = float(F.mse_loss(model(va_x), va_y))
    tr_losses.append(tr); va_losses.append(va)
    if va < best_va:
        best_va, best_ep = va, ep + 1
    print(f"epoch {ep+1:02d}/{EPOCHS}  train={tr:.4f}  val={va:.4f}", flush=True)

print(f"\n最优 epoch = {best_ep}  (val={best_va:.4f})", flush=True)
hist = {"train": [round(v, 4) for v in tr_losses],
        "val": [round(v, 4) for v in va_losses],
        "best_epoch": best_ep, "best_val_loss": round(best_va, 4)}
json.dump(hist, open(os.path.join(RESULTS_DIR, "train_history_35.json"), "w"))
print("已写 train_history_35.json", flush=True)
