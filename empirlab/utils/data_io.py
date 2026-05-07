"""数据读写工具。

提供 CSV / Stata / Parquet 的轻量封装，供模板和 notebook 复用。
"""

from __future__ import annotations

import pandas as pd


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """读取 CSV 文件。"""
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """写出 CSV 文件。"""
    df.to_csv(path, index=index)


def read_dta(path: str, **kwargs) -> pd.DataFrame:
    """读取 Stata .dta 文件。"""
    return pd.read_stata(path, **kwargs)
