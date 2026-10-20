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
"""导出文献对比实验的数据表（CSV + LaTeX），供直接贴进论文。"""
import os
import json

_PROJ = _PKGP.CODE_TSINGHUA
OUT = os.path.join(_PROJ, "results", "lit_compare")
d = json.load(open(os.path.join(OUT, "literature_comparison.json"), encoding="utf-8"))
order = d["order"]
mx = d["matrix"]
s = d["summary"]

METHODS = ["ours_dual", "ijgi", "applsci", "spot"]
M_LABEL = {"ours_dual": "Ours (dual-branch)", "ijgi": "IJGI 2022 (rule)",
           "applsci": "Appl.Sci. 2019 (area)", "spot": "SPOT 2017 (EVT)"}
E_LABEL = {"dwell": "Dwell ($>$5 min)", "geofence": "Geofence intrusion",
           "gather": "Off-hours gathering", "wandering": "Wandering (window-level)",
           "sprint": "Sprinting", "zigzag": "Zigzag"}

# ---- CSV ----
lines = ["event," + ",".join(M_LABEL[m] for m in METHODS)]
for e in order:
    lines.append(E_LABEL[e] + "," + ",".join(str(mx[e][m]) for m in METHODS))
lines.append("coverage," + ",".join(s[m]["coverage"] for m in METHODS))
open(os.path.join(OUT, "literature_comparison.csv"), "w", encoding="utf-8-sig").write("\n".join(lines))

# ---- LaTeX table ----
tex = [
    r"\begin{table}[htbp]",
    r"\caption{Literature comparison: AUROC across six real-trajectory events, all on the same "
    r"Geolife walking trajectories. Ours uses the responsible branch per event (hard rules with a "
    r"15-s dwell ramp for the rule domain, TCN for the motion domain), following the division of "
    r"labour in Table~\ref{tab:ablation}. All events are scored frame-wise except wandering, which "
    r"is scored at window level (the TCN's natural granularity, consistent with Table~6). Best per "
    r"row in bold.}",
    r"\begin{center}",
    r"\begin{tabular}{lcccc}",
    r"\hline",
    r"\textbf{Event} & \textbf{Ours} & \textbf{IJGI\cite{ijgi}} & "
    r"\textbf{Appl.Sci.\cite{applsci}} & \textbf{SPOT\cite{spot}} \\",
    r"\hline",
]
for e in order:
    vals = [mx[e][m] for m in METHODS]
    best = max(vals)
    cells = [E_LABEL[e]]
    for v in vals:
        cells.append((r"\textbf{%.3f}" % v) if v == best else ("%.3f" % v))
    tex.append(" & ".join(cells) + r" \\")
tex += [
    r"\hline",
    " & ".join(["Coverage"] + [s[m]["coverage"] for m in METHODS]) + r" \\",
    r"\hline",
    r"\end{tabular}",
    r"\end{center}",
    r"\end{table}",
]
open(os.path.join(OUT, "table_lit_compare.tex"), "w", encoding="utf-8").write("\n".join(tex))

print("CSV + LaTeX table written to", OUT)
print()
print("\n".join(lines))
print()
print("\n".join(tex))
