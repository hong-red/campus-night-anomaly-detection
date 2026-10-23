# Stanford Drone Dataset —— 强约束 if 规则 + 深度学习 三路消融实验

> 对应清华侧的国外对照实验（`experiment_rule_advantage.py` / `tcn_dl_branch/eval_dwell_wander.py` / `tcn_dl_branch/fusion_ablation.py`）。
> 数据：SDD 真实俯拍行人轨迹（像素 → 米），TCN 在 SDD 正常行走上重训（dt=1s，同架构/同自监督目标）。

---

## 1. 实验1：强约束 if 规则的优势（禁区闯入 + 多人聚集）

**论证**：禁区闯入/多人聚集是「空间 / 跨轨迹共现」属性，不改变单条轨迹的运动模式，
因此单轨迹运动模型（FluxEV motion EVT）对它们无判别力；硬性 if 条件则精确判定。

| 任务 | if 规则 | 规则准确率 | 规则 AUROC | FluxEV 运动 AUROC |
|---|---|---|---|---|
| 禁区闯入（inside-circle） | 68 条真实穿越轨迹 | **accuracy = 1.000** | 1.000 | 0.500（盲） |
| 多人聚集（≥3 人 / 5 m） | 28 条主行人轨迹 | P=0.845 R=1.000 F1=0.916 acc=0.946 | 0.970 | 0.500（盲） |

> 图：`results/sdd_rule_advantage_geofence.png`（轨迹 + 时序）、
> `sdd_rule_advantage_gather.png`（距离 + 时序）、`sdd_rule_advantage_summary.png`（汇总）。
> 数据：`results/sdd_rule_advantage.json`。

---

## 2. 实验2：深度学习（TCN）对「原地逗留 / 徘徊」的准确率·召回率

（TCN 分支自身评估，**窗口级** ADE 作连续分数，@最优 F1 阈值；与主表帧级口径不同，数值不可混用）

| 行为 | AUROC | AP | 精确率 P | 召回率 R | F1 | 准确率 |
|---|---|---|---|---|---|---|
| 徘徊 Wandering（来回折返） | **0.800** | 0.490 | 0.593 | 0.806 | 0.684 | 0.780 |
| 原地逗留 Loitering-in-place（驻留不动） | **0.445** | 0.259 | 0.295 | 1.000 | 0.455 | 0.296 |

> 徘徊偏离「正常人直行」先验 → 预测误差大 → 可检；原地逗留的静止段预测误差近零 →
> 对 TCN 不可见（AUROC<0.5 结构盲区，@最优阈值退化为「全报阳性」）。
> 图：`sdd_dl_roc_pr.png` / `sdd_dl_metrics_bars.png` / `sdd_dl_demo.png`；
> 数据：`sdd_dl_dwell_wander.json`。

---

## 3. 实验3：三路消融 + 表 1（软 + 硬相加效果更好）

四个检测器：`rules`（硬性 if 规则）、`dl`（深度学习 TCN，z-clip 即置信门限）、
`fused = max(rules, dl)`（软+硬相加，核心）、`fluxev`（完整 FluxEV 管线 = EVT 运动 + 规则）。

**表 1：真实校园轨迹上的事件分类、负责分支与 AUROC（7 事件 × 检测器）**

| 事件 | 负责分支 | rules | DL | fused | fluxev |
|---|---|---|---|---|---|
| 长时间停留 Dwell | EVT / if-rule | 0.757 | 0.491 | 0.731 | 0.757 |
| 地理围栏突破 Geofence | EVT / if-rule | **1.000** | 0.543 | 0.987 | 1.000 |
| 非工作时间聚集 Gather | EVT / if-rule | 0.886 | 0.630 | 0.882 | 0.886 |
| 徘徊 Wandering | max(rules, DL) | 0.491 | 0.809 | **0.802** | 0.491 |
| 冲刺 Sprint | max(rules, DL) | 0.495 | 0.834 | **0.829** | 0.495 |
| Zigzag | max(rules, DL) | 0.491 | 0.840 | **0.833** | 0.491 |
| 突然停止 Sudden stop | EVT motion + S_dl 门限 | 0.738 | 0.503 | 0.703 | 0.738 |
| **均值** | — | 0.694 | 0.664 | **0.824** | 0.694 |

### 突停的置信门控（naive max 损害 → 门限恢复）

| 融合方式 | 突停 AUROC |
|---|---|
| S_ev（FluxEV 单独） | 0.738 |
| naive `S = max(S_ev, S_dl)` | 0.703（↓ 损害） |
| gated `max(S_ev, S_dl·[S_dl≥0.5])` | 0.715（↑ 恢复） |

> 图：`sdd_fusion_ablation_bars.png` / `sdd_fusion_ablation_table.png` / `sdd_sudden_stop_gate.png`；
> 数据：`sdd_fusion_ablation.json`。

---

## 4. 结论

1. **规则域事件（长时间停留 / 地理围栏 / 非工作时间聚集）由 EVT（if 规则）分支负责**：
   静止行人、空间越界、跨轨迹共现对 TCN 来说近零预测误差，DL 分支近乎随机（AUROC 0.49~0.63），
   而 if 规则精确命中（0.76~1.00）。
2. **运动模式事件（徘徊 / 冲刺 / zigzag）由两个分支中的最大值负责**：单轨迹 if 规则盲（≈0.49），
   TCN 检出 0.81~0.84，`max(rules, DL)` 继承 TCN 能力（0.80~0.83）。
3. **软性条件 + 硬性条件相加效果更好**：`fused` 均值 AUROC=0.824，显著高于单独硬性规则
   （0.694）、单独深度学习（0.664）与完整 FluxEV 管线（0.694）。
4. **整段取 `S = max(S_ev, S_dl)` 会损害突然停止检测**（0.738 → 0.703），在 `S_dl` 上设置
   置信度门限（`S_dl ≥ 0.5` 才放行）可恢复（0.715）。

---

## 5. 复现

```bash
python sdd_common.py                 # 首次运行会重训 TCN 并落盘 checkpoint
python experiment_rule_advantage_sdd.py   # 实验1：if 规则优势 + 准确率
python eval_dwell_wander_sdd.py           # 实验2：TCN 对原地逗留/徘徊的 P/R
python fusion_ablation_sdd.py             # 实验3：三路消融 + 表1 + 突停门控
python cross_validate_schools.py          # 实验4：两校互为验证集(跨数据集泛化)
```

规则阈值按 SDD 机载俯拍剪辑尺度重标定（判定形式不变，只改时长/半径量纲）：
dwell 停留 >15s（30s 封顶）、gather ≥3 人 / 5 m、geofence 圆形禁区。

---

## 6. 实验4：两所学校互为验证集（跨数据集泛化）

清华侧(Geolife GPS 行走)与斯坦福侧(SDD 俯拍行走)各自训练 TCN(dt=1s，同架构/同目标)，
再互为验证集交叉测试。**帧级 AUROC（主口径，逐帧 ADE，与正文主表一致）**：

| 行为 | Geo→Geo（域内） | **Geo→SDD（迁移）** | SDD→SDD（域内） | **SDD→Geo（迁移）** |
|---|---|---|---|---|
| Wandering 徘徊 | 0.900 | **0.830** | 0.840 | **0.862** |
| Loitering 原地逗留 | 0.112 | **0.457** | 0.479 | **0.109** |
| Sprint 冲刺 | 0.801 | **0.728** | 0.735 | **0.817** |
| Zigzag | 0.806 | **0.794** | 0.813 | **0.793** |
| **均值（含逗留）** | 0.655 | **0.702** | 0.717 | **0.645** |

> 数据：`results/cross_validate_schools_frame.json`（脚本 `cross_validate_schools_frame.py`）。

窗口级（历史口径，仅存档对照，勿与主表混用）：

| 行为 | Geo→Geo（域内） | Geo→SDD（迁移） | SDD→SDD（域内） | SDD→Geo（迁移） |
|---|---|---|---|---|
| Wandering 徘徊 | 0.908 | 0.795 | 0.801 | 0.871 |
| Loitering 原地逗留 | 0.127 | 0.448 | 0.440 | 0.121 |
| Sprint 冲刺 | 0.799 | 0.646 | 0.662 | 0.813 |
| Zigzag | 0.802 | 0.716 | 0.744 | 0.788 |
| **均值** | 0.659 | 0.651 | 0.662 | 0.648 |

结论：
- **迁移 AUROC ≈ 域内 AUROC**（Geo→SDD 0.651≈0.659，SDD→Geo 0.648≈0.662）——
  TCN 学到的「正常人怎么走」先验在两校之间可迁移，互为验证集通过。
- **「原地逗留对 TCN 不可见」是结构盲区，非数据特例**：四个组合 AUROC 全部 < 0.5
  （0.12~0.45），双向一致 → 该盲区只能靠 EVT/规则分支补齐，进一步支持「软+硬相加」。

> 图：`results/cross_validate_schools.png`；数据：`results/cross_validate_schools.json`。
> Geolife 侧 dt=1s 模型：`results/geo_tcn_dt1.pt`（不覆盖 `F:\仿真跑\tcn_dl_branch` 里的 2s checkpoint）。
