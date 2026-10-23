---
name: paper-figures
description: 校园夜间安防论文的出版级 matplotlib 图。当用户要「生图 / 画图 / 做图表 / ROC / PR / 消融图 / 准确率图 / 柱状图 / 曲线图」时使用。统一 Nature 投稿风格（Arial、350dpi、单栏89mm/双栏183mm、面板字母a/b/c、低饱和配色、去顶右脊线），并复用 results/*.json 里的真实数据。
---

# 论文绘图（Nature 风格）

给「校园夜间安防」项目画出版级图。**先读本说明再写图代码**，不要另起一套配色或字号。

## 第一步：统一入口

所有图都复用 `style.py`（与本文件同目录），不要在每个脚本里重写 rcParams：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from style import (apply_nature_style, apply_cn_style, C, BRANCH_COLOR,
                   DOMAIN_COLOR, SINGLE_COL, DOUBLE_COL, MM,
                   panel_label, save, despine, chance_line)
apply_nature_style()          # 投稿版；中文报告/答辩图改用 apply_cn_style()
```

## 风格规范（已固化在 style.py）

- **字体**：Arial / Helvetica / DejaVu Sans；字号 7pt（label 7 / title 7.5 / tick 6.5 / legend 6.2）。
- **尺寸**：单栏 `SINGLE_COL=89mm`、双栏 `DOUBLE_COL=183mm`；高度按内容用 `N*MM` 定。
- **去轴**：无顶/右脊线；线宽 0.6；刻度短（size 2.4）。
- **输出**：350 dpi，`save(fig, name)` 落到 `results/figs/`，自动 `bbox=tight`。
- **面板**：多子图用 `panel_label(ax, "a")` 标字母。
- **随机基线**：`chance_line(ax, 0.5)` 画 0.5 虚线。

## 配色（语义固定，勿换色）

| 用途 | 颜色 | 键 |
|---|---|---|
| 异常/告警 | `#B2182B` 红 | `C["alert"]` |
| 正常/运动/FluxEV | `#4393C3` 蓝 | `C["calm"]` |
| 强调/徘徊 | `#D6604D` 锈橙 | `C["accent"]` |
| 灰(参考线) | `#6E6E6E` | `C["grey"]` |
| 主曲线 | `#252525` 近黑 | `C["dark"]` |
| 规则/围栏 | `#1B7837` 绿 | `C["green"]` |

分支消融（三路硬/软/组合）用 `BRANCH_COLOR`：硬 FluxEV=绿 `#1B7837`、软 TCN=橙 `#eb6834`、组合=黑 `#0b0b0b`、仅规则=灰 `#9AA0A6`。
事件域用 `DOMAIN_COLOR`：规则域=绿、TCN 运动域=橙、EVT 运动(急停)=蓝。

## 数据来源（先读 JSON，勿硬编码）

- 消融 AUROC：`tcn_dl_branch/results/fusion_ablation.json`（7 事件 × 4 检测器）
- 规则域准确率：`tcn_dl_branch/results/rule_accuracy.json`（`per_event[k].specific_rule.acc`）
- TCN 窗口级 P/R/F1（窗口级口径，非主表帧级）：`tcn_dl_branch/results/dl_dwell_wander.json`
- ROC 数据：`tcn_dl_branch/results/fusion_roc_data.json`

## 图表模板

### 1. 横向柱状图（准确率/AUROC/延迟）

```python
fig, ax = plt.subplots(figsize=(SINGLE_COL, 44 * MM))
labels = ["多人聚集", "禁区闯入", "长时间逗留"]      # 中文图用 apply_cn_style
vals = [0.993, 1.000, 0.969]
y = range(len(labels))[::-1]
ax.barh(y, vals, height=0.6, color=C["green"])
for yi, v in zip(y, vals):
    ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=6.5)
chance_line(ax, 0.9, orient="v", color=C["alert"])     # 0.9 达标线
ax.set_yticks(list(y)); ax.set_yticklabels(labels)
save(fig, "accuracy_rules.png")
```

### 2. ROC / PR 双面板

```python
from sklearn import metrics
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 44 * MM))
fpr, tpr, _ = metrics.roc_curve(g, s)
prec, rec, _ = metrics.precision_recall_curve(g, s)
axes[0].plot(fpr, tpr, color=C["alert"]); axes[0].plot([0,1],[0,1], ls="--", color=C["grey"])
axes[0].set_title(f"ROC (AUROC={metrics.roc_auc_score(g,s):.3f})", pad=3)
axes[1].plot(rec, prec, color=C["calm"])
axes[1].set_title(f"PR (AP={metrics.average_precision_score(g,s):.3f})", pad=3)
for a, t in zip(axes, "ab"): panel_label(a, t)
save(fig, "roc_pr.png")
```

### 3. 消融分组柱（7 事件 × 4 检测器）

```python
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 60 * MM))
# x=事件, 每组 4 根柱=BRANCH_COLOR 顺序 fluxev/dl/fused/rules
chance_line(ax, 0.5)
ax.legend(frameon=False)
save(fig, "ablation.png")
```

### 4. 逐帧分数时间序列

```python
fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COL, 70 * MM), sharex=True)
# 上: 各行为分数; 中: 真值 fill_between; 下: final 融合分 + 0.5 阈值线
chance_line(axes[-1], 0.5)
save(fig, "score_timeline.png")
```

## 架构 / 流程图模板（模块块风格，参考手写笔记纵向流程）

用 `style.py` 里的 `DIAGRAM_COLOR` + `diagram_box()` + `diagram_arrow()`，不要手写 FancyBboxPatch。

```python
from style import DIAGRAM_COLOR, diagram_box, diagram_arrow
fig, ax = plt.subplots(figsize=(7.5, 15))       # 纵向流程用高瘦画布
ax.set_xlim(0, 12); ax.set_ylim(0, 24); ax.axis("off")

# 模块块（cx,cy 中心坐标；c 取 input/frozen/train/fusion）
diagram_box(ax, 6.0, 22.2, 5.4, 1.9, ["数据源", "清华 Geolife · 斯坦福 SDD"], "input")
diagram_box(ax, 3.0, 13.6, 4.6, 3.6, ["FluxEV/规则分支（无训练）", "特征提取", "FluxEV 运动评分"], "frozen")
diagram_box(ax, 9.0, 13.6, 4.6, 3.6, ["TCN 分支（可训练）", "轨迹窗口", "预测→ADE"], "train")
diagram_box(ax, 6.0, 9.6, 5.4, 1.5, ["相加融合", "S = max(S_ev, S_dl)"], "fusion")

diagram_arrow(ax, 6.0, 21.25, 6.0, 20.55)       # 直线
diagram_arrow(ax, 6.0, 16.4, 3.0, 15.5, rad=0.15)  # 分叉弧线
```

**配色语义**（`DIAGRAM_COLOR`）：蓝=input 输入/数据/指标、灰=frozen 无训练(FluxEV/规则)、橙=train 可训练(TCN)、绿=fusion 融合/输出。

**纵向流程布局**（对应笔记）：自上而下 数据源 → 轨迹 → 预处理 → 双分支(灰/橙 并列) → 相加融合 → 输出报警 → 评估指标；底部画图例。参考 `tcn_dl_branch\make_vertical_flow.py`。

**横向双分支布局**：输入在左 → 上下两分支 → 右侧融合；参考 `tcn_dl_branch\make_dual_branch_diagram.py`。

## 硬规则

1. **配色/字号只用 style.py 里的**，不自定义 hex 或字号，保证整篇一致。
2. **数据从 results/*.json 读**，不把数字写死进图脚本（否则重跑会过期）。
3. 投稿图用 `apply_nature_style()` + 英文标签 + 面板字母；中文报告图用 `apply_cn_style()` + 中文标签。
4. 输出统一到 `results/figs/`，命名 `fig_<主题>.png`（+ 需要时另存 `.pdf`/`.svg`）。
5. 每张图必画 0.5 随机线或 0.9 达标线这类参考线，并标注关键数值。

## 常用参考

- 现有成品可参照：`02_代码_清华侧\make_nature_figs.py`（fig1–fig6 全套）、`tcn_dl_branch\fusion_ablation.py`（消融条图/表）、`tcn_dl_branch\make_behavior_taxonomy.py`、`tcn_dl_branch\make_rule_accuracy.py`。
- 若做新图：先读对应 results JSON → 写脚本 → `apply_nature_style()` → `save()`。
