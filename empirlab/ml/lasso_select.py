"""LASSO 变量筛选
================
来源参考：
  Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
    *JRSS-B*, 58(1), 267–288.
  Belloni, A., Chernozhukov, V., & Hansen, C. (2014).
    High-dimensional methods and inference on structural and treatment effects.
    *JEP*, 28(2), 29–50.

适用场景：
  - 高维控制变量筛选（"Post-LASSO OLS"）
  - 稳健性检验中的变量重要性排序
  - 机器学习辅助变量选择前处理

方法说明：
  1. LassoCV 自动搜索最优正则化系数 λ
  2. 输出非零系数对应的变量集合（"活跃集"）
  3. 支持 Post-LASSO OLS：用活跃集重新做无偏 OLS
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class LassoSelect:
    """LASSO 变量筛选 + Post-LASSO OLS 估计器。

    参数
    ----
    cv : int, default 5
        交叉验证折数，用于选 λ。
    max_iter : int, default 5000
        LASSO 迭代上限。
    standardize : bool, default True
        是否对 X 标准化后再做 LASSO（推荐）。
    post_ols : bool, default True
        True 时在活跃集上做 Post-LASSO OLS，得到无偏系数。

    属性
    ----
    selected_vars_  : list[str]  —— 被选中的变量名
    alpha_          : float      —— 最优 λ
    lasso_coef_     : pd.Series  —— LASSO 系数（含零）
    post_coef_      : pd.Series  —— Post-OLS 系数（仅活跃集，无偏）

    示例
    ----
    >>> from empirlab.ml import LassoSelect, lasso_data
    >>> df = lasso_data(n=300, p=20, k_true=5)
    >>> y = df.pop("y")
    >>> model = LassoSelect().fit(df, y)
    >>> print("选中变量：", model.selected_vars_)
    """

    def __init__(
        self,
        cv: int = 5,
        max_iter: int = 5000,
        standardize: bool = True,
        post_ols: bool = True,
    ):
        self.cv = cv
        self.max_iter = max_iter
        self.standardize = standardize
        self.post_ols = post_ols

        self.selected_vars_: List[str] = []
        self.alpha_: Optional[float] = None
        self.lasso_coef_: Optional[pd.Series] = None
        self.post_coef_: Optional[pd.Series] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: Optional[List[str]] = None

    def fit(self, X, y) -> "LassoSelect":
        """拟合 LASSO 并可选地做 Post-LASSO OLS。

        参数
        ----
        X : pd.DataFrame 或 array-like, shape (n, p)
        y : array-like, shape (n,)

        返回
        ----
        self
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self._feature_names = [f"x{i+1}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=float).ravel()

        if self.standardize:
            self._scaler = StandardScaler()
            X_fit = self._scaler.fit_transform(X_arr)
        else:
            X_fit = X_arr

        lasso = LassoCV(cv=self.cv, max_iter=self.max_iter, random_state=42)
        lasso.fit(X_fit, y_arr)

        self.alpha_ = float(lasso.alpha_)
        coef = lasso.coef_

        # 如果标准化了，将系数还原到原始量纲
        if self.standardize and self._scaler is not None:
            coef_orig = coef / self._scaler.scale_
        else:
            coef_orig = coef

        self.lasso_coef_ = pd.Series(coef_orig, index=self._feature_names)
        self.selected_vars_ = [
            name for name, c in zip(self._feature_names, coef) if c != 0
        ]

        if self.post_ols and len(self.selected_vars_) > 0:
            if isinstance(X, pd.DataFrame):
                X_sel = X[self.selected_vars_].to_numpy(dtype=float)
            else:
                sel_idx = [self._feature_names.index(v) for v in self.selected_vars_]
                X_sel = X_arr[:, sel_idx]
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_sel, y_arr)
            self.post_coef_ = pd.Series(lr.coef_, index=self.selected_vars_)

        return self

    def summary(self) -> pd.DataFrame:
        """返回变量重要性表（LASSO 系数 + 是否被选中）。"""
        if self.lasso_coef_ is None:
            raise RuntimeError("请先调用 fit()。")
        df = pd.DataFrame({
            "lasso_coef": self.lasso_coef_,
            "selected": self.lasso_coef_ != 0,
        })
        if self.post_coef_ is not None:
            df["post_ols_coef"] = self.post_coef_
        return df.sort_values("lasso_coef", key=abs, ascending=False)


# ── 辅助数据生成 ──────────────────────────────────────────────────────

def lasso_data(
    n: int = 300,
    p: int = 20,
    k_true: int = 5,
    noise: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """生成高维稀疏回归合成数据。

    DGP：只有前 k_true 个变量真正影响 y，其余为噪声变量。

    返回
    ----
    pd.DataFrame，列 x1…xp + y
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    true_coef = np.zeros(p)
    true_coef[:k_true] = rng.uniform(0.5, 2.0, size=k_true) * rng.choice([-1, 1], size=k_true)
    y = X @ true_coef + rng.normal(0, noise, n)
    cols = {f"x{i+1}": X[:, i] for i in range(p)}
    cols["y"] = y
    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = lasso_data(n=400, p=20, k_true=5)
    y = df["y"]
    X = df.drop(columns="y")
    model = LassoSelect(cv=5, post_ols=True).fit(X, y)
    print(f"最优 λ = {model.alpha_:.6f}")
    print(f"选中变量（{len(model.selected_vars_)}个）：{model.selected_vars_}")
    print("\nLASSO 变量重要性：")
    print(model.summary().round(4))
