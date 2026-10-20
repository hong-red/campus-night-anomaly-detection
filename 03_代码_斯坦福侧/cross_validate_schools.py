#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两所学校互为验证集：TCN 轨迹预测分支的跨数据集泛化实验
======================================================

清华侧(F:\\仿真跑，主数据源 Geolife GPS 行走轨迹) 与 斯坦福侧(F:\\Stanford_Drone，
SDD 机载俯拍行人轨迹) 各自训练一个 TCN(同架构/同自监督 MSE，dt=1s)，
再互为验证集交叉测试，回答：

  1) 在 A 校正常行走上学到的"正常人怎么走"先验，能否迁移到 B 校检出异常？
  2) "原地逗留对 TCN 不可见(AUROC<0.5)"是不是方法固有的结构盲区(应双向成立)？

设计(2×2 交叉验证，4 种运动异常 × 4 个 train→test 组合)：
  Geo→Geo   Geolife 训练，Geolife 测试(域内基线)
  Geo→SDD   Geolife 训练，Stanford 测试(迁移)
  SDD→SDD   Stanford 训练，Stanford 测试(域内基线)
  SDD→Geo   Stanford 训练，Geolife 测试(迁移)

输出 (results/)：
  cross_validate_schools.json   4×4 AUROC 矩阵
  cross_validate_schools.png    分组条形图(0.5 随机线)
"""
import os
import json

import sdd_common as S
import tcn_branch as T

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import metrics

RESULTS = S.RESULTS_DIR
GEO_TCN_DT1 = os.path.join(RESULTS, "geo_tcn_dt1.pt")

BEHAVIORS = [("wandering", "Wandering", T.inject_loiter, S.C_WANDER),
             ("loitering", "Loitering (in place)", T.inject_stop, S.C_LOITER),
             ("sprint", "Sprint", T.inject_run, "#eda100"),
             ("zigzag", "Zigzag", T.inject_zigzag, "#1baf7a")]


# --------------------------- 通用训练(任意 (x,y) 轨迹) ---------------------------

def train_tcn_dt1(tracks_xy, ckpt_path, seed=1, max_windows=20000, epochs=10):
    """在 (x,y) 米制轨迹上重训 dt=1s TCN，返回 (model, stats, cfg)。"""
    torch.manual_seed(seed); np.random.seed(seed)
    obs, pred = 16, 8
    obs_arr, fut_arr = T.windowize(tracks_xy, obs, pred, 2, max_windows)
    idx = np.random.default_rng(seed).permutation(len(obs_arr))
    n_val = int(len(obs_arr) * 0.1)
    tr_o, tr_f = obs_arr[idx[n_val:]], fut_arr[idx[n_val:]]
    va_o, va_f = obs_arr[idx[:n_val]], fut_arr[idx[:n_val]]

    model = T.TCNPredictor(in_dim=2, hidden=32, layers=4, k=3, obs_len=obs, pred_len=pred,
                           dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_x = torch.tensor(tr_o, dtype=torch.float32)
    tr_y = torch.tensor(tr_f, dtype=torch.float32)
    va_x = torch.tensor(va_o, dtype=torch.float32)
    va_y = torch.tensor(va_f, dtype=torch.float32)
    best_va, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_x))
        total, cnt = 0.0, 0
        for i in range(0, len(tr_x), 256):
            b = perm[i:i + 256]
            pr = model(tr_x[b])
            loss = torch.nn.functional.mse_loss(pr, tr_y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(b); cnt += len(b)
        va = float(torch.nn.functional.mse_loss(model(va_x), va_y))
        if va < best_va:
            best_va, best_state = va, {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_ade = T.compute_ade(model, va_o, va_f)
    stats = {"mu": float(val_ade.mean()), "sigma": float(val_ade.std()), "offset": 2.0}
    cfg = dict(in_dim=2, hidden=32, layers=4, k=3, obs_len=obs, pred_len=pred, dropout=0.1)
    T.save_checkpoint(ckpt_path, model, stats, cfg)
    print(f"  -> {ckpt_path}  (windows={len(obs_arr)}, mu={stats['mu']:.2f}m, "
          f"sigma={stats['sigma']:.2f}m)")
    return model, stats, cfg


# --------------------------- 窗口级 AUROC ---------------------------

def window_auroc(model, obs, pred, tracks_xy, inject_fn, stride=2):
    ades, labels = [], []
    for (x, y) in tracks_xy:
        n = len(x)
        if n < obs + pred + 24:
            continue
        i0 = int(n * 0.55)
        i1 = min(i0 + max(16, int(n * 0.18)), n - obs - pred - 2)
        if i1 - i0 < 8:
            continue
        xa, ya = x.copy(), y.copy()
        mask = inject_fn(xa, ya, i0, i1)
        obs_list, fut_list, lab_list = [], [], []
        for s in range(0, n - obs - pred + 1, stride):
            fut = np.arange(s + obs, s + obs + pred)
            lab_list.append(1 if np.isin(fut, mask).any() else 0)
            ref = np.array([xa[s + obs - 1], ya[s + obs - 1]])
            obs_list.append(np.stack([xa[s:s + obs], ya[s:s + obs]], -1) - ref)
            fut_list.append(np.stack([xa[s + obs:s + obs + pred],
                                      ya[s + obs:s + obs + pred]], -1) - ref)
        ades.append(T.compute_ade(model, np.asarray(obs_list, np.float32),
                                  np.asarray(fut_list, np.float32)))
        labels.append(np.asarray(lab_list))
    ade = np.concatenate(ades); lab = np.concatenate(labels)
    if lab.sum() in (0, len(lab)):
        return float("nan")
    return float(metrics.roc_auc_score(lab, ade))


# --------------------------- 主流程 ---------------------------

def main():
    obs, pred = 16, 8

    # 两校模型(dt=1s)：SDD 已有；Geolife 补训(不覆盖 tcn_dl_branch 里的 2s checkpoint)
    print("=" * 78)
    print("模型准备(dt=1s)")
    print("=" * 78)
    sdd_model, sdd_stats, sdd_cfg = S.ensure_sdd_tcn()
    print(f"  SDD 模型(训练于 Stanford): {S.SDD_TCN_CKPT}")
    if os.path.exists(GEO_TCN_DT1):
        geo_model, geo_stats, geo_cfg = T.load_predictor(GEO_TCN_DT1)
        print(f"  Geolife 模型(训练于 清华侧): {GEO_TCN_DT1} (已存在)")
    else:
        print("  训练 Geolife TCN(dt=1s) ...")
        geo_traj = T.collect_walking_trajectories(T.DEFAULT_DATA, 1.0, 100, seed=1)
        geo_model, geo_stats, geo_cfg = train_tcn_dt1([(x, y) for x, y, *_ in geo_traj],
                                                      GEO_TCN_DT1)

    # 测试集(互为验证集)
    print("\n" + "=" * 78)
    print("测试集(dt=1s，留出)")
    print("=" * 78)
    sdd_test = [(x, y) for _, x, y, _ in S.collect_sdd_walking(1.0, 100, seed=42, min_len=48)]
    geo_test = [(x, y) for x, y, *_ in T.collect_walking_trajectories(T.DEFAULT_DATA, 1.0,
                                                                      100, seed=42)]
    print(f"  Stanford 测试轨迹 = {len(sdd_test)}，Geolife 测试轨迹 = {len(geo_test)}")

    combos = [("geo_sdd", "Geo→SDD", geo_model, sdd_test),
              ("sdd_geo", "SDD→Geo", sdd_model, geo_test),
              ("geo_geo", "Geo→Geo", geo_model, geo_test),
              ("sdd_sdd", "SDD→SDD", sdd_model, sdd_test)]

    print("\n" + "=" * 78)
    print("交叉验证 AUROC  (4 行为 × 4 train→test 组合)")
    print("=" * 78)
    matrix = {b[0]: {} for b in BEHAVIORS}
    for key, en, inj, _c in BEHAVIORS:
        for ck, cname, model, tracks in combos:
            matrix[key][ck] = round(window_auroc(model, obs, pred, tracks, inj), 3)

    print(f"{'behavior':<22}" + "".join(f"{cname:<12}" for _, cname, _, _ in combos))
    print("-" * 70)
    for key, en, inj, _c in BEHAVIORS:
        print(f"{en:<22}" + "".join(f"{matrix[key][ck]:<12.3f}" for ck, *_ in combos))

    json.dump(matrix, open(os.path.join(RESULTS, "cross_validate_schools.json"), "w"),
              ensure_ascii=False, indent=2)

    # 汇总(迁移 vs 域内)
    print("-" * 70)
    for ck, cname, *_ in combos:
        vals = [matrix[k][ck] for k, *_ in BEHAVIORS]
        print(f"  {cname:<12}  mean AUROC = {np.nanmean(vals):.3f}")

    # 图：分组条形图
    fig, ax = plt.subplots(figsize=(10.0, 3.9))
    x = np.arange(len(BEHAVIORS)); w = 0.19
    combo_color = {"geo_geo": "#9ec5f4", "geo_sdd": "#2a78d6",
                   "sdd_sdd": "#f6c9a8", "sdd_geo": "#eb6834"}
    combo_label = {"geo_geo": "Geo→Geo (in-domain)", "geo_sdd": "Geo→SDD (transfer)",
                   "sdd_sdd": "SDD→SDD (in-domain)", "sdd_geo": "SDD→Geo (transfer)"}
    for j, (ck, cname, *_rest) in enumerate(combos):
        off = -1.5 * w + j * w
        vals = [matrix[k][ck] for k, *_ in BEHAVIORS]
        ax.bar(x + off, vals, w, color=combo_color[ck], zorder=3, label=combo_label[ck])
    ax.axhline(0.5, color="#c3c2b7", lw=1.0, ls="--", zorder=2)
    ax.text(len(BEHAVIORS) - 0.5, 0.505, "chance 0.5", ha="right", va="bottom",
            fontsize=7.5, color="#898781")
    ax.set_xticks(x)
    ax.set_xticklabels([en for _, en, *_ in BEHAVIORS])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Cross-dataset validation of the TCN branch (two schools, mutual test sets)",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=7.5)
    ax.grid(axis="y", color="#e1e0d9", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "cross_validate_schools.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)

    print("\n已保存: results/cross_validate_schools.json / cross_validate_schools.png")


if __name__ == "__main__":
    main()
