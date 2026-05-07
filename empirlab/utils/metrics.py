"""评估指标工具。

这里提供回归任务最常用的几个指标，供模型模板和 notebook 复用。
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def rmse(y_true, y_pred) -> float:
    """计算均方根误差。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true, y_pred) -> float:
    """计算平均绝对误差。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """返回常用回归指标字典。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    rmse_value = float(np.sqrt(mse))
    mae_value = float(np.mean(np.abs(y_true_arr - y_pred_arr)))

    total_var = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    residual_var = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    r2 = 1.0 - residual_var / total_var if total_var > 0 else 0.0

    return {
        "MSE": mse,
        "RMSE": rmse_value,
        "MAE": mae_value,
        "R2": r2,
    }
