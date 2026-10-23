# Stanford Drone Dataset —— 国外校园(斯坦福)场景

对应清华侧（Geolife）的国外对照数据集。

> 全项目（两校）算法/代码/方法总览见 `算法方法总览.md`；本目录实验结果见 `SDD实验结果.md`。

## 内容
- `StanfordDroneDataset/` — SDD 标注(8 场景 × 多视频: `annotations.txt` + `reference.jpg` 场景图)
- `geofence_stanford.json` — 斯坦福大学主校区 OSM 边界(与 `geofence_tsinghua.json` 同格式)
- `run_sdd.py` — 解析 SDD 轨迹(像素→米) → FluxEV + 行为规则, 逐场景出异常效果图
- `run_sdd_rules.py` — 真实 SDD 轨迹上的禁区闯入 / 多人聚集 效果图
- `sdd_common.py` — 共用工具：SDD 数据加载 + 米制 FluxEV 规则管线 + TCN 重训/打分
- `experiment_rule_advantage_sdd.py` — 实验1：强约束 if 规则(禁区闯入/聚集)优势 + 准确率
- `eval_dwell_wander_sdd.py` — 实验2：TCN 对「原地逗留/徘徊」的准确率·召回率
- `fusion_ablation_sdd.py` — 实验3：硬规则 vs 深度学习 vs 相加融合(表1) + 突停置信门控
- `cross_validate_schools.py` — 实验4：两所学校互为验证集(窗口级，存档)
- `cross_validate_schools_frame.py` — 实验4：帧级跨校验证(主口径)
- `SDD实验结果.md` — 四组实验的完整结果与结论
- `results/` — `*_map.png`(轨迹叠参考图) + `*_anomaly.png`(异常分数曲线) + `sdd_*`(实验图)

## 运行
```
python run_sdd.py
python run_sdd_rules.py

# 三路消融实验(首次运行会自动重训 TCN)
python experiment_rule_advantage_sdd.py
python eval_dwell_wander_sdd.py
python fusion_ablation_sdd.py

# 两校互为验证集(会补训 Geolife 侧 dt=1s TCN)
python cross_validate_schools.py
python cross_validate_schools_frame.py
```

## 要点
- 坐标: 像素; 用行人 bbox 主轴(~1.7m 身高)估 m/px(≈0.02~0.05), 速度单位为 m/s。
- 采样: 30fps, 轨迹较长(数百~上万点), FluxEV motion 分支可用(3/3)。
