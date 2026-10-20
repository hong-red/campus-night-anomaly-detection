#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 full_run.log 的满规模 25 epoch 损失解析成 train_history.json，并重画 fig1_loss。
统一掉「fig1_loss 用 20 轮小规模、文档写 25 轮满规模」的不一致。"""
import os
import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOG = os.path.join(RESULTS, "full_run.log")

# 与 make_figures.py 相同的 Nature 风格
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8.0, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "svg.fonttype": "none",
})
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#e1e0d9"

pat = re.compile(r"epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)")
train, val = [], []
for line in open(LOG, encoding="utf-8", errors="ignore"):
    m = pat.search(line)
    if m:
        train.append(float(m.group(2))); val.append(float(m.group(3)))

best_ep = int(min(range(len(val)), key=lambda i: val[i])) + 1
hist = {"train": [round(v, 4) for v in train],
        "val": [round(v, 4) for v in val],
        "best_epoch": best_ep, "best_val_loss": round(val[best_ep - 1], 4)}
json.dump(hist, open(os.path.join(RESULTS, "train_history.json"), "w"))
print(f"解析到 {len(train)} 轮；最优 epoch={best_ep} (val_loss={hist['best_val_loss']})")
print("已写 train_history.json")

# 重画 fig1_loss
fig, ax = plt.subplots(figsize=(3.4, 2.3))
epochs = range(1, len(train) + 1)
ax.plot(epochs, train, color=BLUE, lw=1.4, label="Train")
ax.plot(epochs, val, color=ORANGE, lw=1.4, label="Validation")
ax.scatter([best_ep], [val[best_ep - 1]], s=24, color=ORANGE, zorder=5)
ax.annotate(f"best (ep {best_ep})", xy=(best_ep, val[best_ep - 1]),
            xytext=(best_ep - 6, val[best_ep - 1] + 3), fontsize=6.5, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (m$^2$)")
ax.set_xlim(0.5, len(train) + 0.5)
ax.grid(True, color=GRID, lw=0.5)
ax.legend(frameon=False, loc="upper right")
fig.text(0.005, 0.97, "a", fontsize=10, fontweight="bold", va="top")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig1_loss.png"), bbox_inches="tight")
fig.savefig(os.path.join(RESULTS, "fig1_loss.svg"), bbox_inches="tight")
plt.close(fig)
print("已重画 fig1_loss.png / fig1_loss.svg")
