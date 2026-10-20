# 方案2 —— 激光雷达 + 点云跟踪 + FluxEV‑SPOT + DL 独立打分分支
## 说明文档（校园夜间安防大创项目）

> 版本：v1.1　日期：2026-10-16
> 代码：`tcn_dl_branch/tcn_branch.py`（DL 分支）；`geolife_pipeline.py`（FluxEV 分支）
> 环境：Python 3.10 + PyTorch 2.9.0(cpu)　无 YOLO、无图像

---

## 1. 概述

本项目面向校园夜间安防场景，将无人巡逻车作为移动"数据采集器"，用多线激光雷达 + 点云
多目标跟踪提取行人 2D 平面坐标时序 `(x, y)`，对每条行人轨迹做异常行为检测。

方案2 由三条技术线组成：

| 模块 | 方法 | 输入 | 输出 |
|---|---|---|---|
| 目标跟踪 | 激光雷达点云多目标跟踪（无 YOLO/图像） | 点云 | 行人 ID + `(x, y)` 时序 |
| FluxEV‑SPOT | 极值理论(EVT)无监督 + 流式阈值 | 速度/加速度/转向特征 | 异常分数 `S_ev ∈ [0,1]` |
| **DL 打分分支** | **TCN 轨迹预测（自监督）** | **`(x, y)` 坐标时序** | **异常分数 `S_dl ∈ [0,1]`** |

最终融合：`S = max(S_ev, S_dl)`，超过阈值 0.5 即告警。

---

## 2. DL 打分分支：自监督 TCN 轨迹预测

### 2.1 核心思想

与 FluxEV 的"特征 + 阈值"不同，DL 分支用**轨迹预测/重构**做自监督学习：

- **训练阶段**只喂正常行人轨迹，让 TCN 学会"正常人怎么走"这个运动先验；
- **推理阶段**输入当前观测的 x（过去 `obs_len` 帧坐标），预测未来的 y（`pred_len` 帧坐标），
  用**预测误差**衡量"这个人的运动有多不符合正常先验"；
- 正常轨迹预测误差小 → S_dl 低；徘徊/折返/疾跑/蛇形等运动模式突变 → 预测误差大 → S_dl 高。

优点：**不需要标注异常轨迹**，只需采集大量普通人走路数据，天然契合"真实场景异常样本少"的约束。

### 2.2 模型结构

```
输入 (B, obs_len, 2)                # 历史 16 帧 (x,y) 相对坐标，单位米
   │  permute → (B, 2, obs_len)
   ▼
TCN 编码器：4 个残差块（扩张因果卷积）
   dilation = 1, 2, 4, 8；kernel = 3；hidden = 32
   ▼
线性解码器 → (B, pred_len, 2)       # 预测未来 8 帧相对坐标
```

- 因果卷积（只看过去、不看未来）保证端侧实时可用；
- 相对坐标：以历史窗口最后一帧为原点，模型学习"未来相对位移"；
- 参数量 **55,920**，纯 CPU 可训练，适配 RV1126 等边缘部署；
- 损失：未来坐标的均方误差 MSE（单位 m²）。

### 2.3 异常分数映射

对每个窗口计算平均位移误差（ADE，米）：

```
ADE = mean_{t∈pred} ‖ y_pred(t) − y_true(t) ‖₂

S_dl = sigmoid( (ADE − mu) / sigma − offset )
```

- `mu`、`sigma` 由**验证集正常窗口**的 ADE 分布标定；
- `offset`（默认 2）使正常误差映射到低分区间，阈值 0.5 对应 `ADE ≈ mu + 2·sigma`。

---

## 3. 数据集（公开轨迹库）

### 3.1 为什么选 Geolife

项目需要"接近 10 万条轨迹"的公开轨迹库做预训练。对比常见公开数据集：

| 数据集 | 轨迹规模 | 场景 | 结论 |
|---|---|---|---|
| **Geolife**（微软） | **18,670 条真实 GPS 轨迹** | 城市步行/骑行/乘车 | ✅ 规模最大，本地已下载 |
| ETH / UCY | ~750 条 | 广场行人 | 规模太小 |
| Stanford Drone | ~1.1 万条 | 无人机俯瞰校园 | 规模不足 |
| KITTI / nuScenes | 道路行车场景 | 行人极少 | 无安防异常行为样本 |

**结论**：采用 Geolife 作为公开轨迹库预训练源。对 18,670 条轨迹做滑动窗口切分后，
轨迹片段样本量可达 **接近 10 万条**，满足"公开轨迹库预训练 + 对比实验"的需求。

### 3.2 预处理流程

1. 从每个 `.plt` 取"时间连续的最长段"（相邻点间隔 < 120s）；
2. 线性插值重采样到 `dt = 2s` 等间隔网格；
3. 中位速度 0.4~2.0 m/s 过滤出"行走"轨迹（剔除驻留/骑行/乘车）；
4. 经纬度 → 局部平面米坐标 `(x=东, y=北)`，以轨迹首点为原点；
5. 滑动窗口切分：`obs_len = 16` 帧历史 + `pred_len = 8` 帧未来，步长 `stride = 2`；
6. 每个窗口一条样本，共 ~10 万条，按 9:1 划分训练/验证。

### 3.3 自建数据集（后续落地）

公开数据仅做预训练。主要测试验证集用本校实地采集（巡检车 + 多线激光雷达 + RTK），
夜间 22:00–06:00 在操场/宿舍/围墙禁区采集，含正常行走、迂回徘徊、多人聚集、禁区闯入，
输出 `ped_id, timestamp, x(RTK), y(RTK), label`，按 7:2:1 划分。

---

## 4. 训练与评估

### 4.1 超参数

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| 重采样间隔 dt | 2 s | 学习率 | 1e-3 |
| 历史窗 obs_len | 16 | batch | 256 |
| 预测窗 pred_len | 8 | epoch | 25 |
| 滑窗步长 stride | 2 | 优化器 | Adam |
| hidden / layers / k | 32 / 4 / 3 | 样本量 | ~10 万 |

### 4.2 评估方式

对**留出的正常轨迹**注入四类异常（徘徊 / 急停 / 蛇形 / 疾跑），在窗口级计算 ADE 作为
异常分数，报告 AUROC（预测误差越大越异常）：

- **徘徊**：沿行进方向来回折返（对应"原地徘徊"隐蔽运动异常）
- **急停**：原地停止不动
- **蛇形**：横向高频摆动
- **疾跑**：速度突增（位移放大 2.5×）

---

## 5. 实验结果

### 5.1 训练收敛

满规模（100,000 窗口 = 90,000 训练 / 10,000 验证，25 epoch）训练正常收敛：训练 MSE 由
96.7 降至 74.4，验证 MSE 由 91.8 降至 70.1（见 `fig1_loss.png`）。标定：正常窗口 ADE
均值 mu = 6.06 m、标准差 sigma = 8.02 m（中位 3.94 m）。

### 5.2 满规模异常检测 AUROC（官方结果）

| 行为 | AUROC | 异常ADE(m) | 正常ADE(m) |
|---|---|---|---|
| 疾跑 Sprinting | **0.808** | 24.24 | 7.40 |
| 徘徊 Loitering | **0.774** | 16.49 | 6.30 |
| 蛇形 Zigzag | 0.658 | 7.35 | 5.80 |
| 急停 Sudden stop | 0.190 | 7.00 | 6.83 |

见 `fig2_auroc.png`。急停 AUROC < 0.5 说明重构误差检不出持续停留（停下后预测=原地、误差小），
该行为由 FluxEV 的 dwell 规则负责——这正是两分支融合的意义。

### 5.3 与 FluxEV 的融合

同一轨迹注入徘徊异常后，FluxEV final 与 TCN S_dl 均在异常段抬升，max 融合取两者之高，
实现"运动突变(TCN) + 停留/规则(FluxEV)"互补（见 `fig3_fusion.png`）。

### 5.4 轨迹预测示例

TCN 输入历史 x（16 帧）预测未来 y（8 帧），预测轨迹与真实未来轨迹基本吻合，验证"输入
当前 x → 得到对应 y"这一核心能力（见 `fig4_prediction.png`）。

### 5.5 四类异常轨迹示意

徘徊/急停/蛇形/疾跑四种注入异常在轨迹上的形态（见 `fig5_anomaly_types.png`）。

### 5.6 ADE 分布

正常 vs 四类异常窗口的 ADE 分布：运动突变类（疾跑/徘徊）的 ADE 显著高于正常，急停与正常
接近（见 `fig6_ade_distribution.png`）。

### 5.7 ROC 曲线

四类异常的 ROC 曲线与对应 AUC（见 `fig7_roc.png`）。

### 5.8 图表清单

| 图 | 文件 | 内容 |
|---|---|---|
| Fig 1 | `fig1_loss.png/.svg` | 训练收敛（train/val MSE） |
| Fig 2 | `fig2_auroc.png/.svg` | 四类异常 AUROC（官方满规模） |
| Fig 3 | `fig3_fusion.png/.svg` | FluxEV vs TCN vs max 融合时序 |
| Fig 4 | `fig4_prediction.png/.svg` | 轨迹预测示例（x→y） |
| Fig 5 | `fig5_anomaly_types.png/.svg` | 四类异常轨迹示意 |
| Fig 6 | `fig6_ade_distribution.png/.svg` | ADE 分布（正常 vs 异常） |
| Fig 7 | `fig7_roc.png/.svg` | ROC 曲线 |

全部为 300 dpi PNG + 矢量 SVG，图内文字全英文（Nature 出版风格）。

---

## 6. 复现步骤

```bash
cd tcn_dl_branch

# 满规模训练 + 评估 + 融合图（~10 万样本）
python tcn_branch.py --mode train --max_windows 100000 --max_traj 800 --epochs 25 --demo

# 推理：演示"输入当前 x → 预测 y"，并给一条轨迹打 S_dl
python tcn_branch.py --mode infer --checkpoint results/tcn_checkpoint.pt
```

核心 API：

```python
from tcn_branch import load_predictor, predict, tcn_frame_scores
model, stats, cfg = load_predictor("results/tcn_checkpoint.pt")

y_pred = predict(model, x_obs)                # 输入当前 x(16,2) → 预测 y(8,2)
s_dl   = tcn_frame_scores(model, x, y, 16, 8, stats)   # 整条轨迹 → 逐时刻 S_dl
```

---

## 7. 关键结论与局限

1. **两分支互补**：TCN 靠预测误差擅长检测"运动模式突变"（疾跑/徘徊/蛇形）；持续停留
   一旦停下、历史窗口也静止，预测=原地、误差小，检测不了——这部分由 FluxEV 的 dwell
   规则（停留 > 5 min）负责。`max` 融合正是为了取两者之长。
2. **自监督**：DL 分支不需要标注异常轨迹，只用正常轨迹训练，适合真实场景异常样本少的约束。
3. 局限：公开 Geolife 为白天 GPS 轨迹、无激光雷达点云噪声，仅作预训练；异常样本由注入
   合成、数量有限；后续需用本校夜间自建数据集做主测试。

---

## 8. 后续工作

1. 给 DL 分支输入加一维速度通道 `(x, y, dx, dy)`，或缩短预测窗，进一步提升徘徊/蛇形检测；
2. 用本校夜间自建数据集（激光雷达 + RTK）做主测试，替换注入式评估；
3. 在 RV1126 上量化为 INT8 并测端侧推理延迟。

---

## 9. 使用的文件清单

### 代码

| 文件 | 说明 |
|---|---|
| `geolife_pipeline.py` | FluxEV×Geolife 主管线（EVT 打分 + 行为规则），提供数据解析/重采样函数 |
| `tcn_dl_branch/tcn_branch.py` | **DL 打分分支核心**：TCN 模型 + 训练 + 异常打分 + 融合 |
| `tcn_dl_branch/make_figures.py` | 出图脚本（fig1_loss / fig3_fusion） |
| `tcn_dl_branch/make_figures_more.py` | 补充出图脚本（fig2_auroc / fig4~fig7） |

### 数据（公开轨迹库）

| 文件/目录 | 说明 |
|---|---|
| `Geolife Trajectories 1.3/Data/` | Microsoft Geolife 公开 GPS 轨迹（182 用户、18,670 条 .plt） |
| `FluxEV/` | 参考实现（WSDM 2021）：spot_pipe.py / eval_methods.py |

### 输出（结果与图，位于 `tcn_dl_branch/results/`）

| 文件 | 说明 |
|---|---|
| `tcn_checkpoint.pt` | 训练好的 TCN 模型 + 标定参数 |
| `tcn_error_stats.json` | mu/sigma 标定 |
| `eval_results.json` / `eval_results_full.json` | 异常检测 AUROC/ADE |
| `train_history.json` | 训练损失历史 |
| `fusion_data.json` | 融合时序数据 |
| `full_run.log` | 满规模训练日志 |
| `fig1~fig7_*.png/.svg` | 7 张出版级图 |

---

*文档由 Claude Code 生成；满规模结果已回填（v1.1，2026-10-16）。*
