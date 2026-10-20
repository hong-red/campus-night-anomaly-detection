# 校园夜间安防异常行为检测 · 双分支框架

**Campus Night-time Security Anomaly Detection — A Dual-Branch Trajectory Framework**

> 无人巡逻车 + 多线激光雷达 → 行人 2D 轨迹 `(x, y, t)` → **无 YOLO、无图像**的轨迹级异常行为检测。
> 一套规则 / 极值分支（硬）+ 一套自监督时序预测分支（软），按「负责分支」取 max 融合，覆盖六类夜间异常行为。
> 数据用两个公开数据集替代激光雷达轨迹：**Microsoft Geolife 1.3**（GPS 轨迹）+ **Stanford Drone Dataset**（俯拍行人）。

---

## 一句话结论

把轨迹变成「多维时序特征」，用**两套互补的异常打分器取 max 融合**：

| 分支 | 方法 | 训练 | 覆盖事件 |
|---|---|---|---|
| **EVT / 规则分支**（硬） | FluxEV-SPOT（极值理论无监督）+ dwell / gather / geofence 硬规则 | 无 | 长时间停留 · 地理围栏闯入 · 多人聚集 · 急停 |
| **深度学习分支**（软） | 自监督 TCN 轨迹预测（ADE 误差打分） | 只训正常轨迹 | 徘徊 · 疾跑 · 蛇形 等隐蔽运动异常 |

**核心结论**：规则分支管「规则域」（静止 / 空间 / 跨轨迹共现，对 TCN 不可见）；DL 分支管「预测域」（运动模式，对 FluxEV 不可见）；两者取 **max 融合**（不是平均）效果最好——斯坦福侧均值 **AUROC 0.824**，显著高于单独硬规则 0.694 / 单独深度学习 0.664，且**跨校互为验证集通过**（迁移 ≈ 域内）。

---

## 端到端流程

```
① 数据采集        多线 LiDAR 多目标跟踪 → 行人 2D 轨迹 P = {(x_t, y_t)}
② 预处理          最长连续段 → 均匀重采样 → 中位速度过滤 → 统一米制
                  → 滑窗切分（16 帧历史 + 8 帧未来）→ 正常窗口库
③ 双分支打分      ┌─ EVT/规则分支：运动特征 → FluxEV 波动 + SPOT 定阈 → s_mot
                  │   硬规则：s_dwell / s_gather / s_geo   → S_ev = max(...)
                  └─ TCN 分支：4 层扩张因果卷积预测未来 8 帧 → ADE → S_dl
④ 融合与告警      S = max(S_ev, S_dl · 1[S_dl ≥ 0.5])  （置信门控，保护急停）
⑤ 评估            AUROC / AP / P / R / F1；消融、文献对比、跨校交叉验证
```

---

## 主要结果

### 六行为 · 硬 / 软 / 硬加软三栏（最终主表）

主指标 **AUROC**（0.5 = 随机）。六行为分工：**2× if-else（闯入 / 聚集）+ 1× if-else+EVT（逗留）+ 3× TCN（徘徊 / 疾跑 / 蛇形）**。

| 行为 | 负责分支 | 清华 硬加软 | 斯坦福 硬加软 |
|---|---|---|---|
| Intrusion 闯入 | if-else | **1.000** | **1.000** |
| Gathering 聚集 | if-else | **0.937** | **0.969** |
| Lingering 逗留 | if-else+EVT | **0.974** | **0.762** |
| Wandering 徘徊 | TCN | **0.701** | **0.891** |
| Sprint 疾跑 | TCN | **0.729** | **0.842** |
| Zigzag 蛇形 | TCN | **0.547** | **0.855** |

### 三路消融（核心表）

规则域靠 if 规则、运动域靠 TCN、融合覆盖全部（均值 AUROC）：

| 侧 | 硬规则 | 深度学习 | **融合** |
|---|---|---|---|
| 清华 Geolife | 0.695 | 0.538 | **0.743** |
| 斯坦福 SDD | 0.694 | 0.664 | **0.824** |

### 文献对比（我们 vs 三篇文献，同一批轨迹 / 同一批异常 / 同一 `roc_auc_score`）

6 事件 × 4 方法 AUROC，**每一项都最高**：

| Event | **Ours** | SPOT'17 | IJGI'22 | Appl.Sci.'19 |
|---|---|---|---|---|
| Dwell 逗留 | **0.974** | 0.952 | 0.796 | 0.947 |
| Geofence 闯入 | **1.000** | 0.518 | 0.913 | 0.583 |
| Gather 聚集 | **0.937** | 0.502 | 0.857 | 0.491 |
| Wandering 徘徊 | **0.774** | 0.516 | 0.713 | 0.693 |
| Sprint 疾跑 | **0.729** | 0.624 | 0.437 | 0.462 |
| Zigzag 蛇形 | **0.547** | 0.479 | 0.506 | 0.489 |
| **覆盖率（>0.55）** | **5/6** | 2/6 | 4/6 | 3/6 |

### 跨校交叉验证（两校互为验证集，窗口级 AUROC）

**迁移 AUROC ≈ 域内 AUROC**（0.65 ≈ 0.66）→「正常人怎么走」的先验可跨校迁移；逗留对 TCN 不可见是结构盲区（四个组合全 < 0.5），只能靠 EVT / 规则分支补齐。

> 完整结果、所有表格与图片见 [`README_总览.md`](README_总览.md) 及 `06_消融实验/`、`07_对比实验/`、`08_实验结论/`。

---

## 目录结构

```
campus-night-anomaly-detection/
├── README.md                  ← 本文件（仓库首页）
├── README_总览.md             ← 完整总览：流程 + 代码索引 + 全部实验表 + 复现
├── _paths.py                  ← 自包含路径解析（脚本据此定位包根，克隆即可运行）
├── 00_依赖/                  论文绘图 style.py（paper_figures）
├── 01_文档/                  方法文档、实验结论、参考文献
├── 02_代码_清华侧/           Geolife 侧全部实验代码
│   ├── geolife_pipeline.py   ← 主管线：FluxEV + 行为规则 + max 融合 + 评估
│   ├── FluxEV/               ← FluxEV 参考实现（SPOT / GPD / MOM，WSDM 2021）
│   ├── tcn_dl_branch/        ← TCN 分支 + 消融 / 对比实验
│   ├── 同学思路对比_校准/     ← 最新校准实验（dwell 阈值、硬软相加口径）
│   └── 文献对比/             ← 三篇文献忠实复现 + 对比
├── 03_代码_斯坦福侧/         SDD 侧对照实验 + 跨校交叉验证
├── 04_数据集/                仅 README_数据获取.md（数据未入库，见下）
├── 05_模型权重/              TCN checkpoint × 3
├── 06_消融实验/              消融结果 JSON + 图
├── 07_对比实验/              对比结果 JSON + 图
├── 08_实验结论/              结论汇总 + 最终图
├── 09_电子围栏/              geofence 多边形 JSON × 3
└── 10_论文配图/              论文全部配图（清华侧 / 斯坦福侧）
```

---

## 快速开始

### 环境

- Python 3 + PyTorch + NumPy + pandas + scikit-learn + matplotlib
- 依赖顺序提示（Windows）：先 `import torch` 再 `import numpy/pandas`（规避 c10.dll 冲突，脚本内已处理）

### 数据获取（未入库）

两个公开数据集体量较大，**未随仓库提交**，请按 [`04_数据集/README_数据获取.md`](04_数据集/README_数据获取.md) 下载并放置：

- **Microsoft Geolife 1.3**（约 1.6 GB）→ `04_数据集/Geolife/Data/`
- **Stanford Drone Dataset**（约 445 MB）→ `04_数据集/StanfordDrone/`

所有脚本通过包根 `_paths.py` 定位数据，**克隆到任意机器即可运行**，无需改任何绝对路径。

### 复现

```bash
# 清华侧（Geolife）
python 02_代码_清华侧/geolife_pipeline.py --mode eval
cd 02_代码_清华侧/tcn_dl_branch
python tcn_branch.py --mode train --max_windows 100000 --max_traj 800 --epochs 25 --demo
python eval_dwell_wander.py && python fusion_ablation.py && python compare_methods.py

# 最新校准（逗留 / 硬软相加口径）
python ../同学思路对比_校准/eval_and_plot_final_3cols.py

# 文献对比
python ../文献对比/compare_literature.py && python ../文献对比/export_lit_compare.py

# 斯坦福侧（SDD）
cd ../../03_代码_斯坦福侧
python sdd_common.py                     # 首次运行重训 SDD TCN
python experiment_rule_advantage_sdd.py
python eval_dwell_wander_sdd.py
python fusion_ablation_sdd.py
python cross_validate_schools.py
python eval_fusion_sdd_framewise.py
```

> 完整命令清单见 [`README_总览.md`](README_总览.md) §6。

---

## 关键设计决策

1. **规则阈值随数据尺度重标定**（判定形式不变，只改量纲）：`dwell`、`gather` 等阈值按 Geolife / SDD 的量纲各自标定。
2. **TCN 时间分辨率随数据而定**：Geolife dt=2s、SDD dt=1s；交叉验证统一到 dt=1s。
3. **坐标统一为米制**：经纬度 → 局部平面米；像素 → 米。
4. **融合需置信门控**：朴素 `max(S_ev, S_dl)` 因 S_dl 噪声地板损害「急停」，用 `S_dl·1[S_dl≥0.5]` 门控恢复。
5. **主指标用 AUROC**：异常检测类别严重不平衡，AUROC 基于排序、对阈值不敏感，与 SPOT 等文献口径一致；AP 作辅助。

---

## 特点

- **相机-free / GPU-free / 标注-free**：只依赖激光雷达轨迹，无图像、无检测框标注，可边缘部署。
- **完全可复现**：自包含路径解析 + 数据获取说明，克隆即可运行。
- **跨数据集泛化**：Geolife（GPS）与 SDD（俯拍）互为验证集通过。
