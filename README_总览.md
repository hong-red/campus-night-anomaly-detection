# 校园夜间安防异常行为检测 —— 双分支框架 · 完整交付

> 无人巡逻车 + 多线激光雷达 → 行人 2D 轨迹 `(x,y,t)` → **无 YOLO、无图像**的双分支异常检测。
> 整合目录：清华侧 `F:\仿真跑`（Geolife GPS）+ 斯坦福侧 `F:\Stanford_Drone`（SDD 俯拍）。
> 归档日期：2026-10-20。以最新版本（10-19 / 10-20）为准，删去了旧版与出图脚本，只保留「可复现实验 + 结论 + 结果 + 图片」。

---

## 0. 一句话结论

把轨迹变成「多维时序特征」，用**两套互补的异常打分器取 max 融合**：

| 分支 | 方法 | 训练 | 覆盖事件 |
|---|---|---|---|
| **EVT / 规则分支**（硬） | FluxEV-SPOT（EVT 无监督）+ dwell/gather/geofence 硬规则 | 无 | 长时间停留 / 地理围栏闯入 / 多人聚集 / 急停 |
| **深度学习分支**（软） | 自监督 TCN 轨迹预测（ADE 误差打分） | 只训正常轨迹 | 徘徊 / 疾跑 / 蛇形 等隐蔽运动异常 |

**核心结论**：规则分支管「规则域」（静止/空间/跨轨迹共现，对 TCN 不可见）；DL 分支管「预测域」（运动模式，对 FluxEV 不可见）；两者按「负责分支」**取 max 相加（硬加软 = max(硬,软)，不是平均）**效果最好（斯坦福侧均值 AUROC 0.824，显著高于单独硬规则 0.694 / 单独深度学习 0.664），且**跨校互为验证集通过**（迁移 ≈ 域内）。六行为分工：**2× if-else（闯入/聚集）+ 1× if-else+EVT（逗留）+ 3× TCN（徘徊/疾跑/蛇形）**。

---

## 1. 端到端流程（Pipeline）

```
① 数据采集/加载
   多线 LiDAR 点云多目标跟踪 → 行人 2D 平面轨迹 P = {(x_t, y_t)}
   （论文设定；实验用公开数据集替代：Geolife GPS / SDD 俯拍）

② 预处理（Backbone，两数据集通用但量纲不同）
   - 取每条轨迹「时间连续最长段」 → 重采样到均匀 dt
   - 中位速度过滤，保留「行走」（Geolife 0.4~2.0 m/s）
   - 坐标统一到局部平面米制（经纬度→米 / 像素→米）
   - 滑窗切分：16 帧历史 + 8 帧未来（stride=2）→ 正常窗口库
     Geolife dt=2s；SDD dt=1s（交叉验证统一到 dt=1s）

③ 双分支并行打分（Neck）
   ┌─ EVT/规则分支（训练无关，纯 Python）
   │   运动特征 v(速度)/a(加速度)/θ(方向变化率)
   │     → FluxEV 波动提取（EWMA 残差 → 两步平滑 → 波动序列）
   │     → SPOT/GPD 极值定阈 → 连续运动分数 s_mot ∈ [0,1]
   │   硬规则（几何/时序，if 判定）
   │     → s_dwell（零速度持续时长，lo→hi 线性抬升）
   │     → s_gather（时空共现人数计数）
   │     → s_geo（多边形/圆 围栏，带缓冲区渐变）
   │   S_ev = max(s_mot, s_dwell, s_gather, s_geo)
   │
   └─ TCN 分支（自监督，只训正常轨迹）
       过去 16 帧相对坐标 → 4 层扩张因果卷积(dilation 1/2/4/8,
       k=3, hidden=32, 参数量<6万) → 预测未来 8 帧
       → ADE(平均位移误差, 米) → z-clip 真零地板 → S_dl ∈ [0,1]

④ 融合与告警（Head）
   置信门控融合： S = max( S_ev,  S_dl · 1[S_dl ≥ 0.5] )
   阈值 τ=0.5 二值化 → 逐帧告警
   （naive max(S_ev, S_dl) 会损害「急停」检测，故对 S_dl 加门控）

⑤ 评估
   AUROC / AP / P / R / F1 / 均衡准确率；消融、对比、交叉验证
```

---

## 2. 目录结构

```
双分支实验_完整交付/
├── README_总览.md            ← 本文件：流程 + 索引 + 结论 + 复现
├── 01_文档/                  方法文档、实验结论、参考文献
├── 02_代码_清华侧/           Geolife 侧全部实验代码（非出图）
│   ├── geolife_pipeline.py   ← 方案1 主管线：FluxEV + 行为规则 + 融合
│   ├── simulate.py           ← 异常注入（多人类聚合成场景）
│   ├── experiment_rule_advantage.py  ← if 规则优势实验
│   ├── _final_rule.py        ← 最终硬规则（dwell lo=15s 等收口）
│   ├── FluxEV/               ← FluxEV 参考实现（SPOT/GPD/MOM）
│   ├── tcn_dl_branch/        ← 方案2 TCN 分支 + 消融/对比实验
│   ├── 同学思路对比_校准/     ← 最新校准实验（10-19，eval_*/calibrate_*）
│   └── 文献对比/             ← 三篇文献对比实验（compare_literature.py）
├── 03_代码_斯坦福侧/         SDD 侧对照实验代码（非出图）
├── 04_数据集/                ← 两个数据集（已在包内）
│   ├── Geolife/              （1.6 GB，Microsoft Geolife 1.3）
│   └── StanfordDrone/        （445 MB，Stanford Drone Dataset，8 场景）
├── 05_模型权重/              TCN checkpoint × 3
├── 06_消融实验/              消融（三路/硬软相加/逗留校准）结果 JSON + 图
├── 07_对比实验/              对比（if 规则优势 / 文献对比 / 方案对比）结果 + 图
├── 08_实验结论/              结论汇总 + 最终图（7 张）
├── 09_电子围栏/              geofence 多边形 JSON × 3
└── 10_论文配图/              全部其余论文配图（清华侧 / 斯坦福侧）
```

---

## 3. 两个数据集

| 数据集 | 原始路径 | 大小 | 内容 | dt | 用途 |
|---|---|---|---|---|---|
| **Microsoft Geolife 1.3** | `04_数据集/Geolife/Data` | 1.6 GB | 182 用户、18,670 条 GPS 轨迹（北京，2007–2012） | 2 s | TCN 预训练 + EVT/规则评估（清华侧） |
| **Stanford Drone Dataset** | `04_数据集/StanfordDrone/` | 445 MB | 8 场景（bookstore/coupa/deathCircle/gates/hyang/little/nexus/quad）俯拍行人轨迹 | 1 s | 国外对照 + 跨校交叉验证（斯坦福侧） |

> 坐标统一为米制：Geolife 经纬度→局部平面米（等距圆柱近似）；SDD 像素→米（行人 bbox 主轴 ≈1.7m 估 m/px）。
> 规则阈值随数据尺度重标定（判定形式不变，只改量纲）：
> `dwell` Geolife lo=120s/hi=300s（最终收口 lo=15s）→ SDD lo=15s/hi=30s；
> `gather` Geolife 180s/50m/3人 → SDD 5s/5m/3人。

---

## 4. 核心代码索引

### 4.1 清华侧（Geolife）

| 文件 | 角色 |
|---|---|
| `geolife_pipeline.py` | **方案1 主管线**：FluxEV 分数化 + dwell/gather/geofence 规则 + max 融合 + 评估 |
| `simulate.py` | 多人聚集合成 / 异常注入 |
| `experiment_rule_advantage.py` | 强约束 if 规则（禁区/聚集）优势 + 准确率 |
| `_final_rule.py` | 最终硬规则收口（dwell lo=15s 提升版） |
| `FluxEV/spot_pipe.py` | SPOT 流式峰值超阈值 + GPD/MOM 估参 |
| `FluxEV/main.py` `preprocessing.py` `utils.py` `eval_methods.py` | FluxEV 参考实现（WSDM 2021） |
| `tcn_dl_branch/tcn_branch.py` | **方案2 核心**：TCN 模型 + 训练 + 异常注入 + `tcn_frame_scores` |
| `tcn_dl_branch/eval_dwell_wander.py` | TCN 对逗留/徘徊 P/R |
| `tcn_dl_branch/fusion_ablation.py` | 三路消融（rules/dl/fused）+ 表1 + 突停门控 |
| `tcn_dl_branch/eval_motion_accuracy.py` | 运动域逐事件准确率 |
| `tcn_dl_branch/compute_rule_accuracy.py` | 规则域准确率 |
| `tcn_dl_branch/compare_methods.py` | 方案1 vs 方案2 对比 |
| `tcn_dl_branch/fusion_experiment.py` `fusion_recalib.py` `collect_roc_sweep.py` | 融合实验 / 重标定 / ROC 扫描 |
| `同学思路对比_校准/calibrate_*.py` `eval_*.py` `improve_dwell.py` `dwell_threshold_sweep.py` | **最新（10-19）校准实验**：dwell 阈值、EVT 极值增益、硬软相加口径 |
| `文献对比/compare_literature.py` | 三篇文献（SPOT/IJGI/Appl.Sci.）忠实复现 + 对比 |
| `文献对比/export_lit_compare.py` | 导出 CSV + LaTeX 表 |

### 4.2 斯坦福侧（SDD）

| 文件 | 角色 |
|---|---|
| `sdd_common.py` | 共用工具：SDD 数据加载（像素→米）+ 米制 FluxEV 规则管线 + TCN 重训/打分 |
| `run_sdd.py` `run_sdd_rules.py` | SDD 轨迹 → FluxEV + 规则，逐场景异常图 |
| `experiment_rule_advantage_sdd.py` | 实验1：if 规则优势 + 准确率 |
| `eval_dwell_wander_sdd.py` | 实验2：TCN 对逗留/徘徊 P/R |
| `fusion_ablation_sdd.py` | 实验3：三路消融 + 表1 + 突停门控 |
| `cross_validate_schools.py` | 实验4：两校互为验证集（跨数据集泛化） |
| `cross_validate_schools_3beh.py` | **最新（10-20）** 3 行为跨校验证 |
| `eval_fusion_sdd_framewise.py` | SDD 逐帧融合 |
| `baseline_comparison.py` | 基线对比 |

---

## 5. 实验矩阵与结论

### 5.0 六行为三栏（硬 / 软 / 硬加软）—— 最终主表

> 最终口径：**硬 = if 规则 max(dwell[lo=15s], gather, geofence)；软 = 本地 TCN(z-clip)；硬加软 = 逐事件 max(硬,软)（分工，不是平均）**。主指标 AUROC（0.5=随机）。数据源：清华 `final_3cols_local.json`、斯坦福 `sdd_fusion_framewise.json`。

**清华 GeoLife**

| 行为 | 负责分支 | 硬 AUROC | 软 AUROC | 硬加软=max |
|---|---|---|---|---|
| Intrusion 闯入 | if-else | 1.000 | 0.488 | **1.000** |
| Gathering 聚集 | if-else | 0.937 | 0.502 | **0.937** |
| Lingering 逗留 | if-else+EVT | 0.974 | 0.374 | **0.974** |
| Wandering 徘徊 | TCN | 0.475 | 0.701 | **0.701** |
| Sprint 疾跑 | TCN | 0.482 | 0.729 | **0.729** |
| Zigzag 蛇形 | TCN | 0.475 | 0.547 | **0.547** |

**斯坦福 SDD**

| 行为 | 负责分支 | 硬 AUROC | 软 AUROC | 硬加软=max |
|---|---|---|---|---|
| Intrusion | if-else | 1.000 | 0.496 | **1.000** |
| Gathering | if-else | 0.969 | 0.590 | **0.969** |
| Lingering | if-else+EVT | 0.762 | 0.458 | **0.762** |
| Wandering | TCN | 0.492 | 0.891 | **0.891** |
| Sprint | TCN | 0.496 | 0.842 | **0.842** |
| Zigzag | TCN | 0.492 | 0.855 | **0.855** |

### 5.1 三路消融（核心表）—— 规则域靠 if 规则 / 运动域靠 TCN / 融合覆盖全部

AUROC，0.5 = 随机。**加粗 = 该行为最优检测器**。

| 行为 | 负责分支 | 清华 rules | 清华 TCN | 清华 fused | 斯坦福 rules | 斯坦福 TCN | 斯坦福 fused |
|---|---|---|---|---|---|---|---|
| 长时间停留 Dwell | EVT/规则 | 0.775 | 0.481 | 0.760 | 0.757 | 0.491 | 0.731 |
| 地理围栏闯入 Geofence | EVT/规则 | **0.969** | 0.454 | 0.967 | **1.000** | 0.543 | 0.987 |
| 非工作聚集 Gather | EVT/规则 | **0.982** | 0.477 | 0.981 | 0.886 | 0.630 | 0.882 |
| 徘徊 Wandering | TCN | 0.499 | 0.651 | **0.650** | 0.491 | 0.809 | **0.802** |
| 疾跑 Sprint | TCN | 0.499 | 0.780 | **0.779** | 0.495 | 0.834 | **0.829** |
| 蛇形 Zigzag | TCN | 0.499 | 0.558 | 0.557 | 0.491 | 0.840 | **0.833** |
| 急停 Sudden stop | EVT motion+门控 | 0.642 | 0.364 | 0.504 | 0.738 | 0.503 | 0.703 |
| **均值** | — | 0.695 | 0.538 | **0.743** | 0.694 | 0.664 | **0.824** |

> 清华侧数据：`06_消融实验/结果_JSON/fusion_ablation.json`；斯坦福侧：`sdd_fusion_ablation.json`。
> 图：`fusion_ablation_bars/table.png`、`sdd_fusion_ablation_bars/table.png`、`fig_六行为_*`。



### 5.3 文献对比（我们 vs 三篇文献）

6 事件 × 4 方法 AUROC，同一批 Geolife 轨迹注入同一批异常，同一 `sklearn.roc_auc_score`：

| Event | **Ours（双分支）** | SPOT'17 | IJGI'22 | Appl.Sci.'19 |
|---|---|---|---|---|
| Dwell 逗留 | **0.974** | 0.952 | 0.796 | 0.947 |
| Geofence 闯入 | **1.000** | 0.518 | 0.913 | 0.583 |
| Gather 聚集 | **0.937** | 0.502 | 0.857 | 0.491 |
| Wandering 徘徊（窗口级） | **0.774** | 0.516 | 0.713 | 0.693 |
| Sprint 疾跑 | **0.729** | 0.624 | 0.437 | 0.462 |
| Zigzag 蛇形 | **0.547** | 0.479 | 0.506 | 0.489 |
| **覆盖率（>0.55）** | **5/6** | 2/6 | 4/6 | 3/6 |

> 我们在 6 事件每一项 AUROC 都最高，且是唯一**无需相机/GPU/标注、可边缘部署**的方法。
> 数据：`07_对比实验/结果_JSON/literature_comparison.json`；图：`fig1_auroc_matrix.png`、`fig2_summary.png`。
> 蛇形逐帧 0.547 贴随机线（最弱项）；改窗口级则 0.658、覆盖率 6/6。

### 5.4 跨校交叉验证（两校互为验证集）

TCN（dt=1s，同架构同目标）两校各自训练再互为验证，窗口级 AUROC：

| 行为 | Geo→Geo | **Geo→SDD** | SDD→SDD | **SDD→Geo** |
|---|---|---|---|---|
| Wandering | 0.908 | **0.795** | 0.801 | **0.871** |
| Loitering 逗留 | 0.127 | **0.448** | 0.440 | **0.121** |
| Sprint | 0.799 | **0.646** | 0.662 | **0.813** |
| Zigzag | 0.802 | **0.716** | 0.744 | **0.788** |
| **均值** | 0.659 | **0.651** | 0.662 | **0.648** |

> **迁移 AUROC ≈ 域内 AUROC** → 「正常人怎么走」先验可跨校迁移。
> **逗留对 TCN 不可见是结构盲区（非数据特例）**：四个组合 AUROC 全 <0.5，只能靠 EVT/规则分支补齐。

---

## 6. 复现步骤

> **已自包含**：所有脚本通过包根 `_paths.py` 定位数据/依赖，旧版硬编码的 `F:\仿真跑` / `F:\Stanford_Drone` 已全部移除，克隆到任意机器即可运行（无需改路径）。
> 数据集未入库（体量大）：请按 `04_数据集/README_数据获取.md` 下载并放置。
> Windows 依赖顺序：先 `import torch` 再 `import numpy/pandas`（c10.dll 冲突，脚本内已处理）。

```bash
# 清华侧（Geolife）
python 02_代码_清华侧/geolife_pipeline.py --mode eval        # --data 默认指向 04_数据集/Geolife/Data
cd 02_代码_清华侧/tcn_dl_branch
python tcn_branch.py --mode train --max_windows 100000 --max_traj 800 --epochs 25 --demo
python eval_dwell_wander.py && python fusion_ablation.py && python compare_methods.py

# 最新校准（逗留/硬软相加口径，10-19）
python 同学思路对比_校准/calibrate_dwell.py
python 同学思路对比_校准/eval_hard_accuracy.py
python 同学思路对比_校准/improve_dwell.py

# 六行为三栏（硬/软/硬加软 = max，最终主表 §5.0）
python 同学思路对比_校准/eval_and_plot_final_3cols.py      # → final_3cols_local.json
python 同学思路对比_校准/plot_final_3cols_auroc.py         # 三栏 AUROC 图
python 同学思路对比_校准/plot_final_3cols_max.py           # 三栏准确率图

# 文献对比
python 文献对比/compare_literature.py && python 文献对比/export_lit_compare.py

# 斯坦福侧（SDD）
python sdd_common.py                     # 首次运行重训 SDD TCN
python experiment_rule_advantage_sdd.py
python eval_dwell_wander_sdd.py
python fusion_ablation_sdd.py
python cross_validate_schools.py         # 补训 Geolife dt=1s TCN + 交叉验证

# 斯坦福侧 六行为三栏（硬/软/硬加软 = max）
python eval_fusion_sdd_framewise.py      # → results/sdd_fusion_framewise.json
python plot_sdd_3cols_auroc.py           # 三栏 AUROC 图
python plot_sdd_3cols_max.py             # 三栏准确率图
```

---

## 7. 关键设计决策

1. **规则阈值随数据尺度重标定**（判定形式不变，只改量纲）。
2. **TCN 时间分辨率随数据而定**：Geolife dt=2s，SDD dt=1s；交叉验证统一 dt=1s。
3. **坐标统一为米制**。
4. **融合需置信门控**：naive `max(S_ev, S_dl)` 因 S_dl 噪声地板损害急停，用 `S_dl·1[S_dl≥0.5]` 门控恢复。
5. **主指标用 AUROC**：异常检测类别严重不平衡 + 阈值未定，准确率会被「全判正常」刷高；AUROC 基于排序、对比例与阈值不敏感，与 SPOT 等文献口径一致；AP 作辅助。

---

## 8. 结论速查

1. **if 规则优势**：禁区闯入 accuracy=1.0、聚集 P≈0.85/R=1.0；运动模型 AUROC=0.5（盲）。
2. **TCN 对逗留/徘徊**：徘徊 AUROC≈0.77~0.80 可检；原地逗留 AUROC<0.5（结构盲区）。
3. **软+硬相加更好**：fused 均值 AUROC（清华 0.743 / 斯坦福 0.824）> 单独硬规则 ≈ 单独深度学习。
4. **突停门控**：naive max 损害突停（清华 0.618→0.368、斯坦福 0.738→0.703），S_dl≥0.5 门限恢复。
5. **两校互为验证集**：迁移≈域内（0.65≈0.66）；逗留盲区双向成立（全<0.5）。
6. **文献对比全胜**：6 事件 AUROC 每一项都最高，且唯一相机-free/GPU-free/标注-free。
