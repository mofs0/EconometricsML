"""预处理工具。

这里放最常用的清洗与标准化操作，保证 notebook 里可以直接复用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_impute(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """对数值列进行简单缺失值填补。"""
    result = df.copy()
    numeric_columns = result.select_dtypes(include="number").columns
    for col in numeric_columns:
        if strategy == "mean":
            fill_value = result[col].mean()
        else:
            fill_value = result[col].median()
        result[col] = result[col].fillna(fill_value)
    return result


def standardize(X: np.ndarray) -> np.ndarray:
    """对数组按列做标准化。"""
    X_arr = np.asarray(X, dtype=float)
    mean = X_arr.mean(axis=0, keepdims=True)
    std = X_arr.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X_arr - mean) / std
