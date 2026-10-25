# -*- coding: utf-8 -*-
"""用原生 Excel 图表复刻 cross_validate_schools_3beh.png。

数据源：06_消融实验/结果_JSON/cross_validate_schools_3beh.json
输出：10_论文配图/复刻_Excel/cross_validate_schools_3beh.xlsx
图表：横向分组条形图（4 序列 x 3 行为），可在 Excel/WPS 中双击编辑。
"""
import json
import os

import xlsxwriter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "06_消融实验", "结果_JSON", "cross_validate_schools_3beh.json")
OUT = os.path.join(HERE, "cross_validate_schools_3beh.xlsx")

with open(SRC, encoding="utf-8") as fh:
    data = json.load(fh)

BEHAVIORS = ["wandering", "sprint", "zigzag"]
BEH_LABEL = {"wandering": "Wandering", "sprint": "Sprint", "zigzag": "Zigzag"}
COMBOS = [("geo_geo", "Geo->Geo (in)"),
          ("geo_sdd", "Geo->SDD (trans)"),
          ("sdd_sdd", "SDD->SDD (in)"),
          ("sdd_geo", "SDD->Geo (trans)")]
COLORS = {"geo_geo": "#9EC5F4", "geo_sdd": "#2A78D6",
          "sdd_sdd": "#F6C9A8", "sdd_geo": "#EB6834"}

wb = xlsxwriter.Workbook(OUT)
ws = wb.add_worksheet("data")

title_fmt = wb.add_format({"bold": True, "font_size": 14, "font_name": "Microsoft YaHei"})
hdr_fmt = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1,
                         "align": "center", "font_name": "Microsoft YaHei"})
lbl_fmt = wb.add_format({"bold": True, "font_name": "Microsoft YaHei"})
num_fmt = wb.add_format({"num_format": "0.000", "border": 1, "align": "center",
                         "font_name": "Microsoft YaHei"})

ws.write("A1", "Frame-level cross-dataset generalization (AUROC)", title_fmt)

ws.write("A3", "Behavior", hdr_fmt)
for j, (_, name) in enumerate(COMBOS):
    ws.write(3, 1 + j, name, hdr_fmt)

for i, beh in enumerate(BEHAVIORS):
    r = 4 + i
    ws.write(r, 0, BEH_LABEL[beh], lbl_fmt)
    for j, (key, _) in enumerate(COMBOS):
        ws.write_number(r, 1 + j, float(data[beh][key]), num_fmt)

ws.set_column(0, 0, 14)
ws.set_column(1, 4, 20)
ws.set_row(3, 20)

chart = wb.add_chart({"type": "bar"})
n = len(BEHAVIORS)
for j, (key, name) in enumerate(COMBOS):
    chart.add_series({
        "name":       ["data", 3, 1 + j],
        "categories": ["data", 4, 0, 4 + n - 1, 0],
        "values":     ["data", 4, 1 + j, 4 + n - 1, 1 + j],
        "fill":       {"color": COLORS[key]},
        "border":     {"none": True},
        "data_labels": {"value": True, "num_format": "0.00", "font": {"size": 9}},
    })

chart.set_title({"name": "Frame-level cross-dataset generalization (AUROC)",
                 "name_font": {"size": 13, "bold": True}})
chart.set_x_axis({"name": "AUROC", "min": 0, "max": 1.12, "major_gridlines": {"visible": True},
                  "num_font": {"size": 10}})
chart.set_y_axis({"num_font": {"size": 11}})
chart.set_legend({"position": "top", "font": {"size": 10}})
chart.set_size({"width": 840, "height": 480})
chart.set_chartarea({"border": {"none": True}})

ws.insert_chart("A10", chart)

wb.close()
print("saved", OUT)
