#!/usr/bin/env python3
import os as _os, sys as _sys
_cur = _os.path.dirname(_os.path.abspath(__file__))
for _i in range(6):
    if _os.path.exists(_os.path.join(_cur, "_paths.py")):
        _sys.path.insert(0, _cur)
        break
    _cur = _os.path.dirname(_cur)
import _paths as _PKGP
# -*- coding: utf-8 -*-
"""
把项目"用到的"算法/代码/模型/结果/文档 归档到一个文件夹(只复制不移动，原目录不动)。

数据(Geolife 1.6G、SDD 505M)过大不复制，只在索引里写原始路径。

输出: F:\\Stanford_Drone\\项目整合\\
"""
import os
import shutil

HERE = _PKGP.CODE_STANFORD
TSINGHUA = _PKGP.CODE_TSINGHUA
DST = os.path.join(HERE, "项目整合")


def cp(src, dst_dir, name=None):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name or os.path.basename(src))
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print("  +", dst)


def cp_tree(src_dir, names, dst_dir):
    for n in names:
        cp(os.path.join(src_dir, n), dst_dir)


# 1) 文档
cp_tree(TSINGHUA,
        ["整体方案文档.md", "阶段总结_FluxEV校园安防.txt", "研究方向.txt"],
        os.path.join(DST, "01_文档"))
cp_tree(os.path.join(TSINGHUA, "tcn_dl_branch"),
        ["方案2_DL分支说明.md", "方案对比.md"],
        os.path.join(DST, "01_文档"))
cp_tree(HERE,
        ["SDD实验结果.md", "算法方法总览.md", "参考文献.md", "README.md"],
        os.path.join(DST, "01_文档"))
# 重命名避免两份 README 冲突
os.rename(os.path.join(DST, "01_文档", "README.md"),
          os.path.join(DST, "01_文档", "README_斯坦福.md"))

# 2) 代码_清华侧
c_ts = os.path.join(DST, "02_代码_清华侧")
cp_tree(TSINGHUA, ["geolife_pipeline.py", "experiment_rule_advantage.py", "simulate.py"], c_ts)
cp(os.path.join(TSINGHUA, "FluxEV"), c_ts, "FluxEV")
tcn = os.path.join(c_ts, "tcn_dl_branch")
cp_tree(os.path.join(TSINGHUA, "tcn_dl_branch"),
        ["tcn_branch.py", "eval_dwell_wander.py", "fusion_ablation.py", "compare_methods.py"],
        tcn)

# 3) 代码_斯坦福侧
c_sd = os.path.join(DST, "03_代码_斯坦福侧")
cp_tree(HERE,
        ["sdd_common.py", "run_sdd.py", "run_sdd_rules.py",
         "experiment_rule_advantage_sdd.py", "eval_dwell_wander_sdd.py",
         "fusion_ablation_sdd.py", "cross_validate_schools.py"],
        c_sd)

# 4) 模型
m_dir = os.path.join(DST, "04_模型")
cp(os.path.join(TSINGHUA, "tcn_dl_branch", "results", "tcn_checkpoint.pt"),
   m_dir, "tcn_checkpoint_geolife_dt2s.pt")
cp(os.path.join(HERE, "results", "sdd_tcn_checkpoint.pt"),
   m_dir, "sdd_tcn_checkpoint_dt1s.pt")
cp(os.path.join(HERE, "results", "geo_tcn_dt1.pt"),
   m_dir, "geo_tcn_dt1s.pt")

# 5) 结果(JSON + 关键图)
r_ts_json = os.path.join(DST, "05_结果", "清华侧_JSON")
cp(os.path.join(TSINGHUA, "results", "rule_advantage.json"), r_ts_json)
cp_tree(os.path.join(TSINGHUA, "tcn_dl_branch", "results"),
        ["dl_dwell_wander.json", "fusion_ablation.json", "compare_methods.json",
         "tcn_error_stats.json", "train_history.json"],
        r_ts_json)

r_ts_fig = os.path.join(DST, "05_结果", "清华侧_图")
cp_tree(os.path.join(TSINGHUA, "results"),
        ["rule_advantage_geofence.png", "rule_advantage_gather.png", "rule_advantage_summary.png"],
        r_ts_fig)
cp_tree(os.path.join(TSINGHUA, "tcn_dl_branch", "results"),
        ["dl_roc_pr.png", "dl_metrics_bars.png", "dl_demo.png",
         "fusion_ablation_bars.png", "fusion_ablation_table.png",
         "figc1_auroc.png", "figc2_fusion_time.png"],
        r_ts_fig)

r_sd_json = os.path.join(DST, "05_结果", "斯坦福侧_JSON")
r_sd_fig = os.path.join(DST, "05_结果", "斯坦福侧_图")
for f in os.listdir(os.path.join(HERE, "results")):
    if f.endswith(".json"):
        cp(os.path.join(HERE, "results", f), r_sd_json)
    elif f.startswith("sdd_") and f.endswith(".png"):
        cp(os.path.join(HERE, "results", f), r_sd_fig)

# 6) 电子围栏
gf = os.path.join(DST, "06_电子围栏")
cp_tree(TSINGHUA, ["geofence_bigc.json", "geofence_tsinghua.json"], gf)
cp_tree(HERE, ["geofence_stanford.json"], gf)

print("\n完成:", DST)
