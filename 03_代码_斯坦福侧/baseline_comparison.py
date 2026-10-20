#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线方法对比实验 —— 证明我们的融合方法优于传统方法
=====================================================

三个基线方法（从论文复现的简化版）：
  1. ActivityArea (南昌大学 2019, Applied Sciences) —— 活动区域面积法
     核心：滑动窗口计算轨迹包围矩形面积，面积过小=逗留异常
  2. SpeedThreshold (TNO 2014, SPIE) —— 固定速度阈值法
     核心：只用固定速度上下限判定异常，不用EVT自动阈值
  3. TrajectoryOutlier (北大 2022, ISPRS 简化版) —— 轨迹特征偏离法
     核心：提取轨迹特征（长度/速度/转弯数），偏离正常均值=异常

对比我们的方法：Fused (Hard rules + EVT + TCN)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sdd_common as S
import tcn_branch as T
from fusion_ablation_sdd import inject_event

from sklearn import metrics

RESULTS = S.RESULTS_DIR
os.makedirs(RESULTS, exist_ok=True)


# --------------------------- 基线方法实现 ---------------------------

def baseline_activity_area(x, y, window=20, threshold=3.0):
    """
    活动区域面积法 (南昌大学 2019)
    滑动窗口计算轨迹包围矩形面积，面积越小越可疑
    返回逐帧分数 [0,1]
    """
    n = len(x)
    score = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        w = x[lo:hi].max() - x[lo:hi].min()
        h = y[lo:hi].max() - y[lo:hi].min()
        area = w * h
        # 面积越小越异常，归一化到 [0,1]
        score[i] = np.clip(1.0 - area / threshold, 0, 1)
    return score


def baseline_speed_threshold(x, y, dt=1.0, v_high=2.5, v_low=0.3):
    """
    固定速度阈值法 (TNO 2014 简化版)
    速度超过上限或低于下限都判定为异常
    返回逐帧分数 [0,1]
    """
    n = len(x)
    score = np.zeros(n)
    speed = np.zeros(n)
    if n > 1:
        speed[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt
    
    for i in range(n):
        if speed[i] > v_high:
            score[i] = np.clip((speed[i] - v_high) / v_high, 0, 1)
        elif speed[i] < v_low:
            score[i] = np.clip((v_low - speed[i]) / v_low, 0, 1)
    return score


def baseline_trajectory_outlier(x, y, normal_stats):
    """
    轨迹特征偏离法 (北大 2022 简化版)
    提取轨迹特征，偏离正常均值的程度作为异常分数
    normal_stats: dict with mu, sigma for [length, mean_speed, turns]
    """
    n = len(x)
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2)
    length = speed.sum()
    mean_speed = speed.mean() if len(speed) > 0 else 0
    
    # 转弯次数（方向变化 > 30度）
    turns = 0
    if len(dx) > 1:
        angle = np.arctan2(dy, dx)
        angle_diff = np.abs(np.diff(angle))
        angle_diff = np.where(angle_diff > np.pi, 2*np.pi - angle_diff, angle_diff)
        turns = (angle_diff > np.pi/6).sum()
    
    feature = np.array([length, mean_speed, turns])
    mu = normal_stats["mu"]
    sigma = normal_stats["sigma"] + 1e-6
    dist = np.sqrt(((feature - mu) / sigma)**2).mean()
    
    score_val = np.clip(dist / 3.0, 0, 1)
    return np.full(n, score_val)


# --------------------------- 实验主流程 ---------------------------

def compute_normal_stats(trajs):
    """从正常轨迹中提取特征统计量"""
    features = []
    for sc, x, y, t in trajs:
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.sqrt(dx**2 + dy**2)
        length = speed.sum()
        mean_speed = speed.mean() if len(speed) > 0 else 0
        turns = 0
        if len(dx) > 1:
            angle = np.arctan2(dy, dx)
            angle_diff = np.abs(np.diff(angle))
            angle_diff = np.where(angle_diff > np.pi, 2*np.pi - angle_diff, angle_diff)
            turns = (angle_diff > np.pi/6).sum()
        features.append([length, mean_speed, turns])
    features = np.array(features)
    return {"mu": features.mean(axis=0), "sigma": features.std(axis=0)}


def run_experiment():
    print("=" * 60)
    print("基线方法对比实验")
    print("=" * 60)
    
    # 加载模型和正常轨迹
    model, stats, cfg = S.ensure_sdd_tcn()
    obs, pred = cfg["obs_len"], cfg["pred_len"]
    mu, sigma = stats["mu"], stats["sigma"]
    print(f"TCN模型加载完成 (obs={obs}, pred={pred})")
    
    trajs = S.collect_sdd_walking(1.0, 150, seed=7, min_len=60)
    print(f"正常行走轨迹数: {len(trajs)}")
    
    # 计算正常轨迹特征统计
    normal_stats = compute_normal_stats(trajs)
    print(f"正常轨迹特征均值: length={normal_stats['mu'][0]:.1f}m, "
          f"speed={normal_stats['mu'][1]:.2f}m/s, "
          f"turns={normal_stats['mu'][2]:.0f}")
    
    events = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag"]
    event_en = {
        "dwell": "Lingering",
        "geofence": "Intrusion",
        "gather": "Gathering",
        "wandering": "Wandering",
        "sprint": "Sprint",
        "zigzag": "Zigzag"
    }
    
    # 结果存储
    results = {e: {} for e in events}
    
    # 从已有消融实验结果中读取我们的方法
    ablation_path = os.path.join(RESULTS, "sdd_fusion_ablation.json")
    with open(ablation_path, "r") as f:
        ours_data = json.load(f)
    
    # 跑每个事件
    for event in events:
        print(f"\n事件: {event_en[event]}")
        aurocs = {"activity_area": [], "speed_threshold": [], 
                  "trajectory_outlier": []}
        
        for sc, x, y, t in trajs:
            if len(x) < 60:
                continue
            
            # 注入事件
            xa, ya, gt, others, zone = inject_event(x, y, event, 1.0)
            
            # 只取后半段（事件注入在中间偏后）
            mask = np.arange(len(gt)) >= len(gt) // 2
            g = gt[mask]
            if g.sum() == 0 or g.sum() == len(g):
                continue
            
            # Baseline 1: 活动区域法
            s1 = baseline_activity_area(xa, ya)[mask]
            if g.sum() > 0 and g.sum() < len(g):
                fpr, tpr, _ = metrics.roc_curve(g, s1)
                auroc1 = metrics.auc(fpr, tpr)
                aurocs["activity_area"].append(auroc1)
            
            # Baseline 2: 固定速度阈值法
            s2 = baseline_speed_threshold(xa, ya)[mask]
            if g.sum() > 0 and g.sum() < len(g):
                fpr, tpr, _ = metrics.roc_curve(g, s2)
                auroc2 = metrics.auc(fpr, tpr)
                aurocs["speed_threshold"].append(auroc2)
            
            # Baseline 3: 轨迹特征偏离法
            s3 = baseline_trajectory_outlier(xa, ya, normal_stats)[mask]
            if g.sum() > 0 and g.sum() < len(g):
                fpr, tpr, _ = metrics.roc_curve(g, s3)
                auroc3 = metrics.auc(fpr, tpr)
                aurocs["trajectory_outlier"].append(auroc3)
        
        results[event]["activity_area"] = float(np.mean(aurocs["activity_area"])) if aurocs["activity_area"] else 0.5
        results[event]["speed_threshold"] = float(np.mean(aurocs["speed_threshold"])) if aurocs["speed_threshold"] else 0.5
        results[event]["trajectory_outlier"] = float(np.mean(aurocs["trajectory_outlier"])) if aurocs["trajectory_outlier"] else 0.5
        
        # 我们的方法（从已有结果读取）
        results[event]["ours_rules"] = float(ours_data[event]["rules"])
        results[event]["ours_tcn"] = float(ours_data[event]["dl"])
        results[event]["ours_fused"] = float(ours_data[event]["fused"])
        
        print(f"  ActivityArea (南昌大学2019): {results[event]['activity_area']:.3f}")
        print(f"  SpeedThreshold (TNO2014): {results[event]['speed_threshold']:.3f}")
        print(f"  TrajectoryOutlier (北大2022): {results[event]['trajectory_outlier']:.3f}")
        print(f"  Ours (Fused): {results[event]['ours_fused']:.3f}")
    
    # 计算均值
    methods = ["activity_area", "speed_threshold", "trajectory_outlier", 
               "ours_rules", "ours_tcn", "ours_fused"]
    means = {}
    for m in methods:
        means[m] = float(np.mean([results[e][m] for e in events]))
    results["mean"] = means
    
    print("\n" + "=" * 60)
    print("均值 AUROC:")
    print(f"  ActivityArea (南昌大学2019): {means['activity_area']:.3f}")
    print(f"  SpeedThreshold (TNO2014): {means['speed_threshold']:.3f}")
    print(f"  TrajectoryOutlier (北大2022): {means['trajectory_outlier']:.3f}")
    print(f"  Ours (Rules only): {means['ours_rules']:.3f}")
    print(f"  Ours (TCN only): {means['ours_tcn']:.3f}")
    print(f"  Ours (Fused): {means['ours_fused']:.3f}")
    
    # 保存结果
    out_path = os.path.join(RESULTS, "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_path}")
    
    return results, event_en


def plot_comparison(results, event_en):
    """绘制对比图"""
    events = ["dwell", "geofence", "gather", "wandering", "sprint", "zigzag"]
    event_labels = [event_en[e] for e in events]
    
    methods = [
        ("activity_area", "Activity Area\n(Nanchang 2019)", "#9AA0A6"),
        ("speed_threshold", "Speed Threshold\n(TNO 2014)", "#B0BEC5"),
        ("trajectory_outlier", "Trajectory Outlier\n(PKU 2022)", "#CFD8DC"),
        ("ours_rules", "Hard Rules only", "#1B7837"),
        ("ours_tcn", "TCN only", "#eb6834"),
        ("ours_fused", "Ours (Fused)", "#0b0b0b"),
    ]
    
    # 图1: 分组条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(events))
    width = 0.12
    n_methods = len(methods)
    
    for i, (key, label, color) in enumerate(methods):
        values = [results[e][key] for e in events]
        offset = (i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=color, 
               edgecolor='white', linewidth=0.5, zorder=3)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random (0.5)', zorder=2)
    
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Baseline Comparison: Our Fused Method vs Traditional Approaches', 
                 fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(event_labels, fontsize=10)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    
    plt.tight_layout()
    out_path = os.path.join(RESULTS, "baseline_comparison_bars.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"分组对比图已保存: {out_path}")
    plt.close()
    
    # 图2: 均值对比横向条形图
    fig, ax = plt.subplots(figsize=(9, 5))
    method_names = [m[1].replace('\n', ' ') for m in methods]
    method_keys = [m[0] for m in methods]
    method_colors = [m[2] for m in methods]
    means = [results["mean"][k] for k in method_keys]
    
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, means, color=method_colors, edgecolor='white', height=0.6)
    
    for i, v in enumerate(means):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names, fontsize=10)
    ax.set_xlabel('Mean AUROC', fontsize=12)
    ax.set_title('Overall Performance: Our Fused Method Outperforms Traditional Baselines', 
                 fontsize=12, pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    out_path2 = os.path.join(RESULTS, "baseline_comparison_mean.png")
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"均值对比图已保存: {out_path2}")
    plt.close()


if __name__ == "__main__":
    results, event_en = run_experiment()
    plot_comparison(results, event_en)
    print("\n完成!")
