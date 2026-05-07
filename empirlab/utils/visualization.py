"""可视化工具。

图形由调用方保存或展示，这里只返回 Figure/Axes，不调用 plt.show()。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig, path: str, dpi: int = 300):
    """把 Figure 保存到磁盘。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_actual_vs_pred(y_true, y_pred):
    """绘制真实值与预测值散点图。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true_arr, y_pred_arr, alpha=0.7)
    min_v = float(min(y_true_arr.min(), y_pred_arr.min()))
    max_v = float(max(y_true_arr.max(), y_pred_arr.max()))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    return fig, ax
