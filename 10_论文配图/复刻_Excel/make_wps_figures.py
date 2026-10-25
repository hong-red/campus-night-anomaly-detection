# -*- coding: utf-8 -*-
"""六张图统一风格（WPS/Excel 原生图表）。
规则：Times New Roman 全篇；统一字号(标题12/图例10/数值9/轴10)；统一柱宽(gap=50,overlap=-20)；
统一网格(数值轴主刻度 0.2)；纯白底(图表区+绘图区无边框)；配色用 Image#7 五色板。
输出：图表复刻_WPS.xlsx
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
J = lambda *p: os.path.join(ROOT, *p)

import pandas as pd
import xlsxwriter
from sklearn import metrics

# ---- Image#7 五色板 ----
STEEL, LBLUE, YELLOW, SALMON, OLIVE = "#466F87", "#78B6C8", "#F2D98E", "#E48578", "#98AF1E"
GREY = "#9AA0A6"
FONT = "Times New Roman"
# 统一字号
SZ_TITLE, SZ_LEGEND, SZ_LABEL, SZ_AXIS, SZ_CAT = 12, 10, 9, 10, 9
# 统一几何
GAP, OVL = 50, -20            # 柱宽一致
YGRID = 0.5                   # 数值轴主刻度 0.5（统一分格度）

OUT = os.path.join(ROOT, "图表复刻_WPS.xlsx")
wb = xlsxwriter.Workbook(OUT)
X = lambda **k: wb.add_format({"font_name": FONT, **k})
title_fmt = X(bold=True, font_size=13)
hdr = X(bold=True, bg_color="#F2F2F2", border=1, align="center")
lbl = X(bold=True); num = X(num_format="0.000", border=1, align="center")
note = X(italic=True, font_color="#666666")


def sheet(nm, ncol=9):
    ws = wb.add_worksheet(nm); ws.set_column(0, 0, 18); ws.set_column(1, ncol, 15); return ws


def white(ch):
    ch.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
    ch.set_plotarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})


def base(ch, title, w, h, legend="top"):
    ch.set_title({"name": title, "name_font": {"size": SZ_TITLE, "bold": True, "name": FONT}})
    ch.set_size({"width": w, "height": h})
    ch.set_legend({"position": legend, "font": {"size": SZ_LEGEND, "name": FONT}})
    white(ch)
    ch.set_x_axis({"num_font": {"size": SZ_AXIS, "name": FONT}})
    ch.set_y_axis({"num_font": {"size": SZ_AXIS, "name": FONT}})


def colchart(ws, anchor, series, cats, title, yname, ymin=0.0, ymax=1.0, w=820, h=450):
    ch = wb.add_chart({"type": "column"})
    for s in series:
        ch.add_series({"name": s["name"], "categories": cats, "values": s["vals"],
                       "fill": {"color": s["color"]}, "border": {"none": True},
                       "gap": GAP, "overlap": OVL,
                       "data_labels": {"value": True, "num_format": "0.00",
                                       "font": {"size": SZ_LABEL, "name": FONT}}})
    base(ch, title, w, h)
    ch.set_x_axis({"num_font": {"size": SZ_CAT, "name": FONT}})
    ch.set_y_axis({"name": yname, "min": ymin, "max": ymax, "major_unit": YGRID,
                   "major_gridlines": {"visible": True}, "num_font": {"size": SZ_AXIS, "name": FONT}})
    ws.insert_chart(anchor, ch)
    return ch


def barchart(ws, anchor, series, cats, title, xname, xmax=1.0, w=840, h=430):
    ch = wb.add_chart({"type": "bar"})
    for s in series:
        ch.add_series({"name": s["name"], "categories": cats, "values": s["vals"],
                       "fill": {"color": s["color"]}, "border": {"none": True},
                       "gap": GAP, "overlap": OVL,
                       "data_labels": {"value": True, "num_format": "0.00",
                                       "font": {"size": SZ_LABEL, "name": FONT}}})
    base(ch, title, w, h)
    ch.set_x_axis({"name": xname, "min": 0, "max": xmax, "major_unit": YGRID,
                   "major_gridlines": {"visible": True}, "num_font": {"size": SZ_AXIS, "name": FONT}})
    ch.set_y_axis({"num_font": {"size": SZ_CAT, "name": FONT}})
    ws.insert_chart(anchor, ch)
    return ch


# ========================= 图1 跨数据集泛化 =========================
M = json.load(open(J("06_消融实验", "结果_JSON", "cross_validate_schools_frame.json"), encoding="utf-8"))
ws = sheet("图1_跨数据集泛化")
ws.write("A1", "Frame-level cross-dataset generalization (AUROC)", title_fmt)
behs = [("wandering", "Wandering"), ("sprint", "Sprint"), ("zigzag", "Zigzag")]
combos = [("geo_geo", "Geo→Geo (in-domain)", LBLUE), ("geo_sdd", "Geo→SDD (transfer)", STEEL),
          ("sdd_sdd", "SDD→SDD (in-domain)", YELLOW), ("sdd_geo", "SDD→Geo (transfer)", SALMON)]
ws.write_row(2, 0, ["Behavior"] + [c[1] for c in combos], hdr)
for i, (k, l) in enumerate(behs):
    ws.write(3 + i, 0, l, lbl)
    for j, (ck, *_r) in enumerate(combos):
        ws.write_number(3 + i, 1 + j, float(M[k][ck]), num)
last = 3 + len(behs) - 1
barchart(ws, "A8",
         [{"name": c[1], "vals": ["图1_跨数据集泛化", 3, 1 + j, last, 1 + j], "color": c[2]}
          for j, c in enumerate(combos)],
         ["图1_跨数据集泛化", 3, 0, last, 0],
         "Frame-level cross-dataset generalization (AUROC)", "AUROC", xmax=1.0)

# ========================= 图2 EVT ROC/PR =========================
ev = pd.read_csv(J("02_代码_清华侧", "results", "geolife_scores.csv"))
n = len(ev); tm = np.arange(n) >= n // 2
g = ev["gt_label"].values[tm]; s = ev["final_score"].values[tm]
fpr, tpr, _ = metrics.roc_curve(g, s); prec, rec, _ = metrics.precision_recall_curve(g, s)
auroc = metrics.roc_auc_score(g, s); ap = metrics.average_precision_score(g, s)
ws = sheet("图2_EVT分支ROC_PR")
ws.write("A1", "EVT-branch ROC and PR on the Geolife test segment", title_fmt)
ws.write_row(3, 0, ["ROC fpr", "ROC tpr", "PR recall", "PR precision", "Chance x", "Chance y"], hdr)
nmax = max(len(fpr), len(rec))
for i in range(nmax):
    r = 4 + i
    if i < len(fpr):
        ws.write_number(r, 0, float(fpr[i]), num); ws.write_number(r, 1, float(tpr[i]), num)
    if i < len(rec):
        ws.write_number(r, 2, float(rec[i]), num); ws.write_number(r, 3, float(prec[i]), num)
for i in range(2):
    ws.write_number(4 + i, 4, float(i), num); ws.write_number(4 + i, 5, float(i), num)
last = 3 + nmax
ch = wb.add_chart({"type": "scatter"})
ch.add_series({"name": f"ROC (AUROC {auroc:.3f})", "categories": ["图2_EVT分支ROC_PR", 4, 0, last, 0],
               "values": ["图2_EVT分支ROC_PR", 4, 1, last, 1],
               "marker": {"type": "none"}, "line": {"color": SALMON, "width": 2.25}})
ch.add_series({"name": f"PR (AP {ap:.3f})", "categories": ["图2_EVT分支ROC_PR", 4, 2, last, 2],
               "values": ["图2_EVT分支ROC_PR", 4, 3, last, 3],
               "marker": {"type": "none"}, "line": {"color": STEEL, "width": 2.25}})
ch.add_series({"name": "Random", "categories": ["图2_EVT分支ROC_PR", 4, 4, 5, 4],
               "values": ["图2_EVT分支ROC_PR", 4, 5, 5, 5],
               "marker": {"type": "none"}, "line": {"color": GREY, "width": 2.25, "dash_type": "dash"}})
base(ch, "EVT-branch ROC and PR on the Geolife test segment", 800, 480, "bottom")
ch.set_x_axis({"min": 0, "max": 1, "major_unit": YGRID, "name": "False positive rate / Recall",
               "major_gridlines": {"visible": True}, "num_font": {"size": SZ_AXIS, "name": FONT}})
ch.set_y_axis({"min": 0, "max": 1, "major_unit": YGRID, "name": "True positive rate / Precision",
               "num_font": {"size": SZ_AXIS, "name": FONT}})
ws.insert_chart("A" + str(last + 3), ch)
ws.write(last + 3, 8, f"AUROC={auroc:.3f}  AP={ap:.3f}", note)

# ========================= 图3 三分支对比（2x2） =========================
B7 = ["Intrusion", "Gathering", "Lingering", "Wandering", "Sprint", "Zigzag", "Sudden stop"]
blocks = [
    ("Tsinghua GeoLife — Accuracy", "Accuracy",
     [1, .97, .78, .5, .5, .5, .68], [.78, .62, .54, .62, .68, .63, .61], [1, .97, .78, .62, .68, .63, .68]),
    ("Tsinghua GeoLife — AUROC", "AUROC",
     [1, .94, .97, .47, .48, .47, .75], [.49, .5, .37, .70, .73, .55, .34], [1, .94, .97, .70, .73, .55, .75]),
    ("Stanford Drone — Accuracy", "Accuracy",
     [1, .95, .65, .5, .5, .5, .62], [.68, .69, .57, .81, .82, .77, .61], [1, .95, .65, .81, .82, .77, .62]),
    ("Stanford Drone — AUROC", "AUROC",
     [1, .97, .76, .49, .5, .49, .74], [.5, .59, .46, .89, .84, .85, .48], [1, .97, .76, .89, .84, .85, .74]),
]
ws = sheet("图3_三分支对比")
ws.write("A1", "EVT vs TCN vs EVT+TCN across 7 behaviours", title_fmt)
for bi, (bname, ylab, evt, tcn, fus) in enumerate(blocks):
    r0 = 3 + bi * 12
    ws.write(r0, 0, bname, X(bold=True, font_size=11))
    ws.write_row(r0 + 1, 0, ["Behavior", "EVT", "TCN", "EVT+TCN"], hdr)
    for i, b in enumerate(B7):
        ws.write(r0 + 2 + i, 0, b, lbl)
        ws.write_number(r0 + 2 + i, 1, evt[i], num)
        ws.write_number(r0 + 2 + i, 2, tcn[i], num)
        ws.write_number(r0 + 2 + i, 3, fus[i], num)
    cats = ["图3_三分支对比", r0 + 2, 0, r0 + 2 + len(B7) - 1, 0]
    ser = [{"name": "EVT", "vals": ["图3_三分支对比", r0 + 2, 1, r0 + 2 + 6, 1], "color": STEEL},
           {"name": "TCN", "vals": ["图3_三分支对比", r0 + 2, 2, r0 + 2 + 6, 2], "color": SALMON},
           {"name": "EVT+TCN", "vals": ["图3_三分支对比", r0 + 2, 3, r0 + 2 + 6, 3], "color": OLIVE}]
    anchor = ["A52", "H52", "A76", "H76"][bi]
    colchart(ws, anchor, ser, cats, bname, ylab, 0.0, 1.15, w=620, h=400)

# ========================= 图4 文献对比 =========================
ws = sheet("图4_文献对比")
ws.write("A1", "Literature comparison — AUROC across 7 real-trajectory events", title_fmt)
lit = [("Lingering", .97, .95, .37, .36), ("Intrusion", 1, .52, .48, .46), ("Gathering", .94, .5, .48, .47),
       ("Wandering", .70, .51, .44, .39), ("Sprint", .73, .62, .55, .51), ("Zigzag", .55, .48, .44, .39),
       ("Sudden stop", .75, .67, .30, .31)]
meth = ["Ours (dual-branch)", "SPOT'17 (EVT)", "GRADINGS'20 (NF)", "ICHCI'21 (AE)"]
mcol = [SALMON, STEEL, LBLUE, OLIVE]
ws.write_row(2, 0, ["Event"] + meth, hdr)
for i, row in enumerate(lit):
    ws.write(3 + i, 0, row[0], lbl)
    for j in range(4):
        ws.write_number(3 + i, 1 + j, float(row[1 + j]), num)
last = 3 + len(lit) - 1
colchart(ws, "A14",
         [{"name": meth[j], "vals": ["图4_文献对比", 3, 1 + j, last, 1 + j], "color": mcol[j]}
          for j in range(4)],
         ["图4_文献对比", 3, 0, last, 0],
         "Literature comparison — AUROC across 7 real-trajectory events", "AUROC", 0.0, 1.15, w=900, h=460)

# ========================= 图5 融合策略对比 =========================
ws = sheet("图5_融合策略对比")
ws.write("A1", "Fusion strategy, naive max vs gated max (c=0.5)", title_fmt)
fus = [("Wandering", .50, .67, .51), ("Sprint", .51, .79, .62),
       ("Zigzag", .50, .61, .50), ("Sudden stop", .62, .37, .62)]
fs = ["S_ev only", "Naive max", "Gated c=0.5"]
fcol = [STEEL, SALMON, OLIVE]
ws.write_row(2, 0, ["Behavior"] + fs, hdr)
for i, row in enumerate(fus):
    ws.write(3 + i, 0, row[0], lbl)
    for j in range(3):
        ws.write_number(3 + i, 1 + j, float(row[1 + j]), num)
last = 3 + len(fus) - 1
colchart(ws, "A10",
         [{"name": fs[j], "vals": ["图5_融合策略对比", 3, 1 + j, last, 1 + j], "color": fcol[j]}
          for j in range(3)],
         ["图5_融合策略对比", 3, 0, last, 0],
         "Fusion strategy, naive max vs gated max (c=0.5)", "AUROC", 0.0, 1.0, w=820, h=460)
ws.write(last + 3, 0, "Sudden stop: naive max 0.62→0.37, gated max 恢复至 0.62", note)

# ========================= 图6 TCN 训练曲线 =========================
th = json.load(open(J("06_消融实验", "结果_JSON", "train_history_35.json"), encoding="utf-8"))
ws = sheet("图6_TCN训练曲线")
ws.write("A1", "TCN training: optimal at epoch 25", title_fmt)
ws.write_row(2, 0, ["Epoch", "Train Loss", "Val Loss"], hdr)
for i, (tr, va) in enumerate(zip(th["train"], th["val"])):
    ws.write_number(3 + i, 0, i + 1, num)
    ws.write_number(3 + i, 1, float(tr), num); ws.write_number(3 + i, 2, float(va), num)
last = 3 + len(th["train"]) - 1
ch = wb.add_chart({"type": "line"})
ch.add_series({"name": "Train Loss", "categories": ["图6_TCN训练曲线", 3, 0, last, 0],
               "values": ["图6_TCN训练曲线", 3, 1, last, 1], "marker": {"type": "none"},
               "line": {"color": SALMON, "width": 2.25}})
ch.add_series({"name": "Val Loss", "categories": ["图6_TCN训练曲线", 3, 0, last, 0],
               "values": ["图6_TCN训练曲线", 3, 2, last, 2], "marker": {"type": "none"},
               "line": {"color": STEEL, "width": 2.25}})
base(ch, "TCN training: optimal at epoch 25", 820, 460, "bottom")
ch.set_x_axis({"name": "Epoch", "major_gridlines": {"visible": True},
               "num_font": {"size": SZ_AXIS, "name": FONT}})
ch.set_y_axis({"name": "MSE Loss (m$^2$)", "min": 60, "max": 100, "major_unit": 10,
               "major_gridlines": {"visible": True}, "num_font": {"size": SZ_AXIS, "name": FONT}})
ws.insert_chart("A" + str(last + 3), ch)
ws.write(last + 3, 5, "最优 Epoch=25 / Val MSE=70.06；Train 97→72（epoch 35）", note)

wb.close()
print("saved ->", OUT)

