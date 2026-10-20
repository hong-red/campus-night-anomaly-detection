# 数据集获取说明

> 两个公开数据集体量较大（Geolife 约 1.6 GB / StanfordDrone 约 445 MB），
> 未随仓库提交。请按下面步骤下载并放到对应目录后，即可直接复现。

## 目录结构（下载后应满足）

```
04_数据集/
├── Geolife/
│   └── Data/
│       ├── 000/Trajectory/*.plt
│       ├── 001/Trajectory/*.plt
│       └── ...（182 个用户目录，共 18670 个 .plt）
└── StanfordDrone/
    ├── bookstore/video0/annotations.txt
    ├── coupa/...
    ├── deathCircle/...
    ├── gates/...
    ├── hyang/...
    ├── little/...
    ├── nexus/...
    └── quad/...
```

## 下载来源

### Microsoft Geolife 1.3（清华侧主数据源）
- 官网：https://www.microsoft.com/en-us/download/details.aspx?id=52367
- 或 ResearchGate / 学术镜像搜索「Geolife GPS Trajectories 1.3」。
- 解压后把 `Data/` 目录放到 `04_数据集/Geolife/Data/`。

### Stanford Drone Dataset（斯坦福侧对照）
- 官网：https://cvgl.stanford.edu/projects/uav_data/
- 下载 8 个场景（bookstore/coupa/deathCircle/gates/hyang/little/nexus/quad）的
  `video*/annotations.txt`（每场景 `video*` 目录内的 `annotations.txt` + `reference.jpg`）。
- 放到 `04_数据集/StanfordDrone/`。

## 路径自检

放好后运行（应打印 `True`）：

```bash
python -c "import _paths as P, os; print(os.path.isdir(P.GEOLIFE_DATA), os.path.isdir(P.SDD_DATA))"
```

> 无需任何绝对路径：所有脚本通过包根 `_paths.py` 定位数据集，
> 旧版硬编码的 `F:\仿真跑` / `F:\Stanford_Drone` 已全部移除。
