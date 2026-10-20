# -*- coding: utf-8 -*-
"""
包内自包含路径解析（本文件必须位于包根目录）。

所有脚本通过「向上查找本文件」的方式定位包根，从而不再依赖任何绝对路径
（如旧的 F:\\仿真跑 / F:\\Stanford_Drone），使本仓库可直接克隆到任意机器复现。
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集
GEOLIFE_DATA = os.path.join(ROOT, "04_数据集", "Geolife", "Data")      # Microsoft Geolife 1.3
SDD_DATA = os.path.join(ROOT, "04_数据集", "StanfordDrone")            # Stanford Drone Dataset

# 代码
CODE_TSINGHUA = os.path.join(ROOT, "02_代码_清华侧")
CODE_STANFORD = os.path.join(ROOT, "03_代码_斯坦福侧")
TCN_DIR = os.path.join(CODE_TSINGHUA, "tcn_dl_branch")

# 依赖 / 权重
STYLE_DIR = os.path.join(ROOT, "00_依赖", "paper_figures")             # style.py（论文绘图）
WEIGHTS_DIR = os.path.join(ROOT, "05_模型权重")
