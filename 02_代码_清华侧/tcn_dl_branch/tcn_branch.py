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
"""
TCN 轨迹预测分支（方案2 的独立深度学习打分分支，无 YOLO、无图像）
=================================================================

定位
----
与 FluxEV(EVT 无监督) 并列的第二个异常打分器。输入是行人 2D 轨迹坐标时序 (x, y)，
不是图像。用【轨迹预测/重构】做自监督训练：只喂正常轨迹，让 TCN 学"正常人怎么走"，
推理时用"预测误差"作为异常分数 S_dl ∈ [0,1]。异常行为（原地徘徊/来回折返/急停/
疾跑/突然转向）偏离学到的正常运动先验 → 预测误差大 → S_dl 高。

两个核心 API（对应用户的两个诉求）
--------------------------------
    predict(model, x_obs)                 # 输入当前 x(obs_len×2 相对坐标) → 预测 y(pred_len×2)
    tcn_frame_scores(model, x, y, ...)    # 整条轨迹 → 每个时刻的 S_dl ∈ [0,1]

数据集（公开轨迹库，接近 10 万条轨迹片段）
------------------------------------------
主数据源 = 本地已下载的 Microsoft Geolife（182 用户、18670 个 .plt 文件、真实 GPS 轨迹）。
    - 从每个 .plt 取"时间连续的最长段"，重采样到 dt 秒；
    - 用中位速度 0.4~2.0 m/s 过滤出"行走"轨迹（非驻留、非骑行/乘车）；
    - 经纬度 → 局部平面米坐标 (x=东, y=北)，以每条轨迹首点为原点；
    - 滑动窗口切分：obs_len 帧历史 + pred_len 帧未来，每个窗口一条样本；
    - 切出的样本量 = 接近 10 万条（--max_windows 100000 可开满）。
说明：ETH/UCY(~750 条)、Stanford Drone(~1.1 万条) 规模都远小于 Geolife，
      Geolife 是本项目最贴合的"公开轨迹库"预训练源。

模型
----
TCN(扩张因果卷积) 编码器 + 线性解码器：
    input  (B, obs_len, 2)  ->  TCN(dilations=1,2,4,8)  ->  (B, pred_len, 2)
Loss = 均方误差(未来坐标, 米)。纯 CPU 可训练，模型参数 < 5 万。

异常分数映射
------------
    ADE = 预测未来坐标与真实未来坐标的平均欧氏距离(米)
    S_dl = sigmoid( (ADE - mu) / sigma - offset )
    mu/sigma 由验证集正常窗口的 ADE 分布标定，offset 默认 2(正常误差→低分)。

运行
----
    # 快速演示(子集)：训练 → 标定 → 注入异常评估 → 与 FluxEV 融合对比图
    python tcn_branch.py --mode train --demo

    # 满规模：接近 10 万条轨迹片段
    python tcn_branch.py --mode train --max_windows 100000 --max_traj 800 --epochs 25

    # 推理：加载 checkpoint，演示"输入当前 x → 得到预测 y"，并给一条轨迹打 S_dl
    python tcn_branch.py --mode infer --checkpoint results/tcn_checkpoint.pt

依赖：torch / numpy / pandas / scipy / matplotlib / scikit-learn（复用上一级 geolife_pipeline 的解析函数）。
"""

import os
import sys
import glob
import argparse
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)                 # 上一级 = F:\仿真跑（geolife_pipeline.py、Geolife 数据在此）
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# 注意：Windows 下 numpy/pandas 的 OpenMP/MKL 运行库与 torch 的 c10.dll 冲突，
# 必须先 import torch（让 torch 先加载 libiomp5md），再 import numpy/pandas，
# 否则报 OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败。
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 复用现有管线的数据解析/特征函数（模块级会顺手把 FluxEV 挂进 sys.path，无副作用）
from geolife_pipeline import (parse_plt, longest_continuous_segment, resample,
                              trajectory_features, V_STATIONARY, MAX_STATIC_FRAC,
                              run_pipeline)

# 中文字体（与 geolife_pipeline 相同处理）
for _fp in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc"):
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATA = _PKGP.GEOLIFE_DATA
RESULTS_DIR = os.path.join(_HERE, "results")
CKPT_PATH = os.path.join(RESULTS_DIR, "tcn_checkpoint.pt")


# ============================ 数据准备 ============================

def latlon_to_local(lat, lon):
    """经纬度 → 局部平面米坐标 (x=东, y=北)，以首点为原点（等距圆柱近似，校园尺度误差可忽略）。"""
    lat0, lon0 = lat[0], lon[0]
    x = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110540.0
    return x, y


def local_to_latlon(x, y, lat0, lon0):
    """局部米坐标 → 经纬度（FluxEV 分支需要 lat/lon 输入）。"""
    lat = lat0 + y / 110540.0
    lon = lon0 + x / (111320.0 * np.cos(np.radians(lat0)))
    return lat, lon


def collect_walking_trajectories(data_dir, dt, max_traj, seed, min_len=60, min_dur=300):
    """
    扫描 Geolife Data 目录，返回行走轨迹列表。
    每条: (x, y, lat, lon, t, path)，x/y 为局部米坐标(首点为原点)。
    """
    plt_files = sorted(glob.glob(os.path.join(data_dir, "*", "Trajectory", "*.plt")))
    rng = np.random.default_rng(seed)
    rng.shuffle(plt_files)
    trajs = []
    for p in plt_files:
        parsed = parse_plt(p)
        if parsed is None:
            continue
        lats, lons, t = parsed
        if t[-1] - t[0] < min_dur:
            continue
        lats, lons, t = longest_continuous_segment(lats, lons, t)
        if len(lats) < min_len:
            continue
        lat, lon, tg = resample(lats, lons, t, dt)
        if len(lat) < min_len:
            continue
        speed, _, _ = trajectory_features(lat, lon, dt)
        med = float(np.median(speed))
        p90 = float(np.percentile(speed, 90))
        frac_stat = float((speed < V_STATIONARY).mean())
        if not (0.4 <= med <= 2.0 and p90 < 4.0 and frac_stat <= MAX_STATIC_FRAC):
            continue
        x, y = latlon_to_local(lat, lon)
        trajs.append((x, y, lat, lon, tg, p))
        if len(trajs) >= max_traj:
            break
    return trajs


def windowize(trajs, obs_len, pred_len, stride, max_windows):
    """
    把行走轨迹切成 (历史窗口, 未来窗口) 样本对。
    相对坐标：以历史窗口最后一帧为原点，所以模型学的是"未来相对位移"。
    返回 (obs_arr, fut_arr)，shape = (N, obs_len, 2) / (N, pred_len, 2)，单位米。
    """
    obs_list, fut_list = [], []
    total = obs_len + pred_len
    for x, y, *_rest in trajs:
        n = len(x)
        for s in range(0, n - total + 1, stride):
            ref = np.array([x[s + obs_len - 1], y[s + obs_len - 1]])
            obs = np.stack([x[s:s + obs_len], y[s:s + obs_len]], axis=-1) - ref
            fut = np.stack([x[s + obs_len:s + total], y[s + obs_len:s + total]], axis=-1) - ref
            obs_list.append(obs)
            fut_list.append(fut)
            if len(obs_list) >= max_windows:
                return np.asarray(obs_list, dtype=np.float32), np.asarray(fut_list, dtype=np.float32)
    return np.asarray(obs_list, dtype=np.float32), np.asarray(fut_list, dtype=np.float32)


# ============================ TCN 模型 ============================

class CausalConv1d(nn.Module):
    """因果卷积（只看过去，不看未来）。"""

    def __init__(self, in_c, out_c, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_c, out_c, k, dilation=dilation, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        if self.pad:
            out = out[:, :, :-self.pad]
        return out


class TCNBlock(nn.Module):
    """TCN 残差块（两层扩张因果卷积 + 残差）。"""

    def __init__(self, in_c, out_c, k, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv1d(in_c, out_c, k, dilation)
        self.c2 = CausalConv1d(out_c, out_c, k, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        y = self.drop(self.act(self.c1(x)))
        y = self.c2(y)
        return self.act(y + self.res(x))


class TCNPredictor(nn.Module):
    """TCN 编码器 + 线性解码器：输入历史坐标时序，输出未来坐标时序。"""

    def __init__(self, in_dim=2, hidden=32, layers=4, k=3, obs_len=16, pred_len=8, dropout=0.1):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        blocks = []
        for i in range(layers):
            in_c = in_dim if i == 0 else hidden
            blocks.append(TCNBlock(in_c, hidden, k, 2 ** i, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * obs_len, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, pred_len * in_dim),
        )

    def forward(self, x):
        # x: (B, obs_len, in_dim)
        x = x.permute(0, 2, 1)              # (B, in_dim, obs_len)
        h = self.tcn(x)                     # (B, hidden, obs_len)
        out = self.head(h)                  # (B, pred_len*in_dim)
        return out.view(-1, self.pred_len, 2)


# ============================ 训练 / 推理 / 打分 ============================

def compute_ade(model, obs_arr, fut_arr, batch=2048):
    """批量计算每个窗口的 ADE(平均位移误差, 米)。"""
    model.eval()
    ades = []
    with torch.no_grad():
        for i in range(0, len(obs_arr), batch):
            ox = torch.tensor(obs_arr[i:i + batch], dtype=torch.float32, device=DEVICE)
            fx = torch.tensor(fut_arr[i:i + batch], dtype=torch.float32, device=DEVICE)
            pred = model(ox)
            err = torch.norm(pred - fx, dim=-1).mean(dim=-1)   # (B,)
            ades.append(err.cpu().numpy())
    return np.concatenate(ades)


def ade_to_score(ade, mu, sigma, offset=2.0):
    """把 ADE(米) 单调映射到 [0,1]：正常误差→低分，异常误差→高分。"""
    z = (ade - mu) / (sigma + 1e-8) - offset
    return 1.0 / (1.0 + np.exp(-z))


def predict(model, x_obs):
    """
    核心 API ①：输入当前观测 x_obs，得到预测的 y。

    x_obs : (obs_len, 2) 相对坐标(米，以最后一帧为原点) 或 (B, obs_len, 2)
    返回  : (pred_len, 2) 预测未来相对坐标(米)
    """
    model.eval()
    arr = np.asarray(x_obs, dtype=np.float32)
    single = arr.ndim == 2
    if single:
        arr = arr[None]
    with torch.no_grad():
        pred = model(torch.tensor(arr, dtype=torch.float32, device=DEVICE))
    pred = pred.cpu().numpy()
    return pred[0] if single else pred


def tcn_frame_scores(model, x, y, obs_len, pred_len, stats, stride=1):
    """
    核心 API ②：整条轨迹 → 每个时刻的 S_dl ∈ [0,1]（长度与输入一致，头部/尾部无窗口处为 NaN）。

    x, y  : 局部米坐标序列(绝对坐标)
    stats : dict，含 mu/sigma/offset
    """
    mu, sigma, offset = stats["mu"], stats["sigma"], stats.get("offset", 2.0)
    n = len(x)
    total = obs_len + pred_len
    scores = np.full(n, np.nan)
    if n < total:
        return scores
    for s in range(0, n - total + 1, stride):
        ref = np.array([x[s + obs_len - 1], y[s + obs_len - 1]])
        obs = np.stack([x[s:s + obs_len], y[s:s + obs_len]], axis=-1) - ref
        fut = np.stack([x[s + obs_len:s + total], y[s + obs_len:s + total]], axis=-1) - ref
        ade = float(compute_ade(model, obs[None], fut[None])[0])
        sc = float(ade_to_score(ade, mu, sigma, offset))
        scores[s + obs_len] = sc
    return scores


def save_checkpoint(path, model, stats, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "stats": stats,
        "cfg": cfg,
    }, path)


def load_predictor(path):
    ck = torch.load(path, map_location=DEVICE)
    cfg = ck["cfg"]
    model = TCNPredictor(in_dim=cfg["in_dim"], hidden=cfg["hidden"], layers=cfg["layers"],
                         k=cfg["k"], obs_len=cfg["obs_len"], pred_len=cfg["pred_len"],
                         dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ck["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, ck["stats"], cfg


# ============================ 异常注入（评估用） ============================

def inject_loiter(x, y, i0, i1, amp=8.0, cycle=16):
    """徘徊：沿行进方向来回折返(原地往复)。"""
    cx, cy = float(np.mean(x[i0:i1])), float(np.mean(y[i0:i1]))
    dx, dy = x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]
    norm = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / norm, dy / norm
    idx = np.arange(i0, i1)
    phase = np.sin(2 * np.pi * (idx - i0) / cycle)
    x[i0:i1] = cx + ux * amp * phase
    y[i0:i1] = cy + uy * amp * phase
    return np.arange(i0, i1)


def inject_stop(x, y, i0, i1):
    """急停：原地不动。"""
    x[i0:i1] = x[i0]
    y[i0:i1] = y[i0]
    return np.arange(i0, i1)


def inject_zigzag(x, y, i0, i1, amp=5.0, cycle=20):
    """蛇形/折返：横向高频摆动。"""
    dx, dy = x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]
    norm = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux                       # 垂直方向
    idx = np.arange(i0, i1)
    osc = amp * np.sin(2 * np.pi * (idx - i0) / cycle)
    x[i0:i1] += px * osc
    y[i0:i1] += py * osc
    return np.arange(i0, i1)


def inject_run(x, y, i0, i1, factor=2.5):
    """疾跑：位移放大(速度突增)。"""
    dx, dy = x - x[i0], y - y[i0]
    x[i0:i1] = x[i0] + dx[i0:i1] * factor
    y[i0:i1] = y[i0] + dy[i0:i1] * factor
    return np.arange(i0, i1)


# ============================ 主流程 ============================

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("扫描 Geolife 行走轨迹 ...")
    trajs = collect_walking_trajectories(args.data, args.dt, args.max_traj, args.seed)
    if not trajs:
        print("未找到可用行走轨迹，请检查 --data 路径")
        sys.exit(1)
    print(f"收集到 {len(trajs)} 条行走轨迹，切分窗口 ...")
    obs_arr, fut_arr = windowize(trajs, args.obs, args.pred, args.stride, args.max_windows)
    print(f"样本窗口数 = {len(obs_arr)}  (目标 ~10 万: 设 --max_windows 100000)")

    # 训练/验证划分
    idx = np.random.default_rng(args.seed).permutation(len(obs_arr))
    n_val = int(len(obs_arr) * 0.1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr_obs, tr_fut = obs_arr[tr_idx], fut_arr[tr_idx]
    val_obs, val_fut = obs_arr[val_idx], fut_arr[val_idx]
    print(f"训练 {len(tr_idx)} / 验证 {len(val_idx)}")

    model = TCNPredictor(in_dim=2, hidden=args.hidden, layers=args.layers, k=args.k,
                         obs_len=args.obs, pred_len=args.pred, dropout=args.dropout).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 = {n_param}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_x = torch.tensor(tr_obs, dtype=torch.float32, device=DEVICE)
    tr_y = torch.tensor(tr_fut, dtype=torch.float32, device=DEVICE)
    va_x = torch.tensor(val_obs, dtype=torch.float32, device=DEVICE)
    va_y = torch.tensor(val_fut, dtype=torch.float32, device=DEVICE)

    tr_losses, va_losses = [], []
    best_va, best_ep, best_state = float("inf"), -1, None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_x))
        total, cnt = 0.0, 0
        for i in range(0, len(tr_x), args.batch):
            b = perm[i:i + args.batch]
            pred = model(tr_x[b])
            loss = F.mse_loss(pred, tr_y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(b); cnt += len(b)
        tr_loss = total / cnt
        va_loss = float(F.mse_loss(model(va_x), va_y))
        tr_losses.append(tr_loss); va_losses.append(va_loss)
        print(f"epoch {ep+1:02d}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        # 最优 checkpoint：按验证集 loss 取 argmin，证明「验证集最优」
        if va_loss < best_va:
            best_va, best_ep = va_loss, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 恢复验证集 loss 最低的那一轮权重（而非最后一轮）
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"最优 epoch = {best_ep}/{args.epochs}  (val_loss = {best_va:.4f})")

    # 标定：用验证集正常窗口的 ADE 分布
    val_ade = compute_ade(model, val_obs, val_fut)
    stats = {"mu": float(val_ade.mean()), "sigma": float(val_ade.std()), "offset": args.offset}
    print(f"标定: mu(ADE)={stats['mu']:.3f}m  sigma={stats['sigma']:.3f}m  "
          f"(正常窗口 ADE 中位 {np.median(val_ade):.3f}m)")

    cfg = dict(in_dim=2, hidden=args.hidden, layers=args.layers, k=args.k,
               obs_len=args.obs, pred_len=args.pred, dropout=args.dropout)
    save_checkpoint(CKPT_PATH, model, stats, cfg)
    json.dump(stats, open(os.path.join(RESULTS_DIR, "tcn_error_stats.json"), "w"))
    # 训练历史统一落盘（供 fig1_loss 与论文复现，含最优 epoch）
    hist = {"train": [round(v, 4) for v in tr_losses],
            "val": [round(v, 4) for v in va_losses],
            "best_epoch": best_ep, "best_val_loss": round(best_va, 4)}
    json.dump(hist, open(os.path.join(RESULTS_DIR, "train_history.json"), "w"))

    # 损失曲线
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(tr_losses, label="train loss (MSE)")
    ax.plot(va_losses, label="val loss (MSE)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (m2)"); ax.set_title("TCN trajectory-prediction training curve")
    ax.legend(); fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "tcn_train.png"), dpi=130); plt.close(fig)

    # 注入异常评估（窗口级 AUROC）
    print("\n" + "=" * 72)
    print("注入异常评估（窗口级，ADE 作为异常分数，AUROC）")
    print("=" * 72)
    injectors = {"徘徊": inject_loiter, "急停": inject_stop, "蛇形": inject_zigzag, "疾跑": inject_run}
    eval_trajs = collect_walking_trajectories(args.data, args.dt, args.max_traj, args.seed + 999)
    from sklearn import metrics
    total = args.obs + args.pred
    summary = []
    for name, fn in injectors.items():
        ades, labels = [], []
        for (x, y, *_rest) in eval_trajs:
            n = len(x)
            if n < total + 24:
                continue
            i0 = int(n * 0.55)
            i1 = min(i0 + max(16, int(n * 0.18)), n - total - 2)
            if i1 - i0 < 8:
                continue
            xa, ya = x.copy(), y.copy()
            mask = fn(xa, ya, i0, i1)
            obs_list, fut_list, lab_list = [], [], []
            for s in range(0, n - total + 1, args.stride):
                fut_seg = np.arange(s + args.obs, s + total)
                lab_list.append(1 if mask[np.isin(mask, fut_seg)].size else 0)
                ref = np.array([xa[s + args.obs - 1], ya[s + args.obs - 1]])
                obs_list.append(np.stack([xa[s:s + args.obs], ya[s:s + args.obs]], -1) - ref)
                fut_list.append(np.stack([xa[s + args.obs:s + total], ya[s + args.obs:s + total]], -1) - ref)
            ades.append(compute_ade(model, np.asarray(obs_list, np.float32),
                                    np.asarray(fut_list, np.float32)))
            labels.append(np.asarray(lab_list))
        a = np.concatenate(ades); l = np.concatenate(labels)
        auroc = metrics.roc_auc_score(l, a) if l.sum() not in (0, len(l)) else float("nan")
        summary.append((name, auroc, float(a[l == 1].mean()), float(a[l == 0].mean())))
        print(f"  {name:<4} AUROC={auroc:.3f}  异常ADE={a[l==1].mean():.2f}m  正常ADE={a[l==0].mean():.2f}m")
    if args.demo:
        demo_fusion(model, stats, args)
    return model, stats


def demo_fusion(model, stats, args):
    """端到端演示：同一轨迹上 FluxEV 分数 vs TCN S_dl，再 max 融合。"""
    print("\n生成 FluxEV × TCN 融合对比图 ...")
    trajs = collect_walking_trajectories(args.data, args.dt, args.max_traj, args.seed + 7)
    # 选一条够长、且第二段速度不静止的轨迹
    traj = None
    for t in trajs:
        x, y, lat, lon, tg, _ = t
        if len(x) >= 160:
            traj = t
            break
    if traj is None:
        print("没有足够长的轨迹做融合演示，跳过")
        return
    x, y, lat, lon, tg, _ = traj
    n = len(x)
    i0, i1 = int(n * 0.55), min(int(n * 0.75), n - args.obs - args.pred - 2)
    xa, ya = x.copy(), y.copy()
    mask = inject_loiter(xa, ya, i0, i1)

    # TCN S_dl（逐帧）
    s_dl = tcn_frame_scores(model, xa, ya, args.obs, args.pred, stats, stride=1)

    # FluxEV final（需要 lat/lon 输入）
    lat_a, lon_a = local_to_latlon(xa, ya, lat[0], lon[0])
    res = run_pipeline(lat_a, lon_a, tg, args.dt, others=None, zone=None, poly=None, q=0.005)
    s_ev = res["final"]

    # 融合（NaN 视为 0）
    s_dl_f = np.nan_to_num(s_dl, nan=0.0)
    fused = np.maximum(s_ev, s_dl_f)

    gt = np.zeros(n); gt[i0:i1] = 1
    tt = tg
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(xa, ya, lw=0.6, color="0.4")
    axes[0].scatter(xa[i0:i1], ya[i0:i1], s=3, color="red")
    axes[0].set_title("Trajectory (red = injected loitering anomaly)"); axes[0].set_ylabel("y (m)")
    axes[1].plot(tt, s_ev, lw=1.1, color="#0B6BFF"); axes[1].set_ylabel("FluxEV")
    axes[2].plot(tt, s_dl_f, lw=1.1, color="#00B140"); axes[2].set_ylabel("TCN S_dl")
    axes[3].plot(tt, fused, color="#E8272C", lw=1.2, label="max fusion")
    axes[3].fill_between(tt, 0, 1, where=gt > 0, color="red", alpha=0.15)
    axes[3].axhline(0.5, color="#333333", ls="--", lw=0.9); axes[3].set_ylabel("Fused score"); axes[3].legend(fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(); os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "tcn_fluxev_fusion.png"), dpi=130); plt.close(fig)
    print("已保存 results/tcn_fluxev_fusion.png")


def infer(args):
    model, stats, cfg = load_predictor(args.checkpoint)
    obs_len, pred_len = cfg["obs_len"], cfg["pred_len"]

    # 演示核心 API ①：输入当前 x → 预测 y
    rng = np.random.default_rng(args.seed)
    trajs = collect_walking_trajectories(args.data, args.dt, args.max_traj, args.seed)
    if not trajs:
        print("无可用轨迹"); return
    x, y, lat, lon, tg, path = trajs[0]
    total = obs_len + pred_len
    s = rng.integers(0, max(1, len(x) - total))
    ref = np.array([x[s + obs_len - 1], y[s + obs_len - 1]])
    x_obs = np.stack([x[s:s + obs_len], y[s:s + obs_len]], -1) - ref
    y_true = np.stack([x[s + obs_len:s + total], y[s + obs_len:s + total]], -1) - ref
    y_pred = predict(model, x_obs)
    err = float(np.linalg.norm(y_pred - y_true, axis=-1).mean())
    print("\n核心 API ① 演示：输入当前 x → 预测 y")
    print(f"  输入 x ({obs_len} 帧历史相对坐标, 米) 最后 3 帧:\n{np.round(x_obs[-3:], 2)}")
    print(f"  预测 y ({pred_len} 帧未来, 米):\n{np.round(y_pred, 2)}")
    print(f"  真实 y ({pred_len} 帧, 米):\n{np.round(y_true, 2)}")
    print(f"  平均位移误差 ADE = {err:.3f} m")

    # 核心 API ②：整条轨迹打分
    s_dl = tcn_frame_scores(model, x, y, obs_len, pred_len, stats, stride=1)
    valid = s_dl[~np.isnan(s_dl)]
    print(f"\n核心 API ② 演示：整条轨迹 S_dl（{len(valid)} 个有效时刻）")
    print(f"  S_dl 分位数 [50,90,95,99]% = {np.round(np.percentile(valid, [50,90,95,99]), 3)}")
    print(f"  S_dl >= 0.5 的时刻占比 = {(valid >= 0.5).mean()*100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="TCN 轨迹预测异常打分分支")
    ap.add_argument("--mode", choices=["train", "infer"], default="train")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", default=CKPT_PATH)
    # 数据
    ap.add_argument("--dt", type=float, default=2.0, help="重采样间隔(秒)")
    ap.add_argument("--obs", type=int, default=16, help="历史窗口帧数")
    ap.add_argument("--pred", type=int, default=8, help="预测未来帧数")
    ap.add_argument("--stride", type=int, default=2, help="滑窗步长(帧)")
    ap.add_argument("--max_traj", type=int, default=250, help="最多采集多少条行走轨迹")
    ap.add_argument("--max_windows", type=int, default=20000, help="最多切多少窗口(满规模设 100000)")
    # 模型/训练
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--offset", type=float, default=2.0, help="ADE→分数映射的偏置")
    ap.add_argument("--demo", action="store_true", help="训练后生成 FluxEV×TCN 融合图")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
