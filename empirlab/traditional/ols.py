"""普通最小二乘回归 (OLS)
========================
来源参考：Greene, W. H. (2003). Econometric Analysis (5th ed.), Chapter 3.
适用场景：线性回归基准估计，含完整统计推断（标准误、t 检验、F 检验、稳健 SE）。
Python 版本：3.10+
依赖：numpy >= 1.24, pandas >= 1.5, scipy >= 1.10
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from empirlab.utils.metrics import calculate_metrics


class OLS:
    """普通最小二乘回归 (Ordinary Least Squares).

    估计线性模型 y = X @ beta + eps，eps ~ N(0, sigma^2 * I).
    提供经典同方差标准误与 HC1 稳健标准误两套推断。

    参数
    ----
    fit_intercept : bool, default True
        是否自动加入截距项（常数列）。
    robust : bool, default False
        True 时 summary() 使用 HC1 异方差稳健标准误。

    属性
    ----
    intercept_ : float
    coef_       : np.ndarray, shape (n_features,)
    feature_names_ : list[str] or None
    is_fitted   : bool

    示例
    ----
    >>> import numpy as np
    >>> from empirlab.traditional import OLS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 3))
    >>> y = X @ [1.5, -0.8, 0.3] + rng.standard_normal(300) * 0.5
    >>> m = OLS(robust=True).fit(X, y)
    >>> print(m.summary_table().round(4))
    """

    def __init__(self, fit_intercept: bool = True, robust: bool = False):
        self.fit_intercept = fit_intercept
        self.robust = robust
        self.is_fitted: bool = False
        self.intercept_: float = 0.0
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self._X_design: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._beta_full: Optional[np.ndarray] = None

    # ── 内部工具 ──────────────────────────────────────────────────────

    def _to_numpy(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _ols_se(self) -> np.ndarray:
        """经典 OLS 标准误：SE = sqrt(sigma^2 * diag((X'X)^-1))."""
        n, k = self._X_design.shape
        rss = float(np.sum(self._residuals ** 2))
        sigma2 = rss / (n - k)
        XtX_inv = np.linalg.pinv(self._X_design.T @ self._X_design)
        return np.sqrt(sigma2 * np.diag(XtX_inv))

    def _hc1_se(self) -> np.ndarray:
        """HC1 异方差稳健标准误（White 1980，小样本修正版）."""
        n, k = self._X_design.shape
        e = self._residuals
        XtX_inv = np.linalg.pinv(self._X_design.T @ self._X_design)
        Xe = self._X_design * e[:, None]
        meat = Xe.T @ Xe
        sandwich = XtX_inv @ meat @ XtX_inv
        return np.sqrt(np.diag((n / (n - k)) * sandwich))

    def _inference(self, se: np.ndarray) -> dict:
        n, k = self._X_design.shape
        df = n - k
        t = self._beta_full / se
        p = 2 * stats.t.sf(np.abs(t), df=df)
        cv = stats.t.ppf(0.975, df=df)
        return dict(t=t, p=p, lo=self._beta_full - cv * se, hi=self._beta_full + cv * se)

    def _f_test(self) -> tuple:
        n, k = self._X_design.shape
        k_reg = (k - 1) if self.fit_intercept else k
        if k_reg == 0:
            return np.nan, np.nan
        ess = float(np.sum((self._y_pred - self._y_true.mean()) ** 2))
        rss = float(np.sum(self._residuals ** 2))
        if rss == 0:
            return np.nan, np.nan
        f = (ess / k_reg) / (rss / (n - k))
        return float(f), float(stats.f.sf(f, k_reg, n - k))

    # ── 公开接口 ──────────────────────────────────────────────────────

    def fit(self, X, y) -> "OLS":
        """拟合 OLS 模型。

        参数
        ----
        X : array-like or pd.DataFrame, shape (n, p)
        y : array-like, shape (n,)

        返回
        ----
        self（支持链式调用）
        """
        X_arr = self._to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        if X_arr.ndim != 2:
            raise ValueError("X 必须是二维矩阵。")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X 与 y 行数不一致。")

        ones = np.ones((X_arr.shape[0], 1))
        self._X_design = np.hstack([ones, X_arr]) if self.fit_intercept else X_arr
        self._beta_full = np.linalg.pinv(self._X_design) @ y_arr

        if self.fit_intercept:
            self.intercept_ = float(self._beta_full[0])
            self.coef_ = self._beta_full[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._beta_full

        self._y_true = y_arr
        self._y_pred = self.predict(X_arr)
        self._residuals = y_arr - self._y_pred
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """线性预测。"""
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()。")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.intercept_ + X_arr @ self.coef_

    def residuals(self) -> np.ndarray:
        """返回训练残差 e = y - y_hat。"""
        if self._residuals is None:
            raise RuntimeError("请先调用 fit()。")
        return self._residuals

    def summary(self) -> Dict[str, object]:
        """返回完整计量经济学摘要字典。

        键说明
        ------
        model, se_type, n_obs, n_features, df_model, df_resid
        coefficients : dict[param_name -> {coef, se, t, p_value, ci_lower, ci_upper}]
        fit          : {R2, adj_R2, F_stat, F_pval, sigma, MSE, RMSE, MAE}
        residuals    : {mean, std, min, max}
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")

        n, k = self._X_design.shape
        k_reg = (k - 1) if self.fit_intercept else k

        se = self._hc1_se() if self.robust else self._ols_se()
        inf = self._inference(se)
        f_stat, f_pval = self._f_test()

        metrics = calculate_metrics(self._y_true, self._y_pred)
        r2 = metrics["R2"]
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k_reg - 1) if n > k_reg + 1 else np.nan
        sigma = float(np.sqrt(np.sum(self._residuals**2) / (n - k)))

        names = self.feature_names_ or [f"x{i+1}" for i in range(k_reg)]
        pnames = (["const"] + names) if self.fit_intercept else names

        coef_tbl = {
            nm: dict(
                coef=float(self._beta_full[i]),
                se=float(se[i]),
                t=float(inf["t"][i]),
                p_value=float(inf["p"][i]),
                ci_lower=float(inf["lo"][i]),
                ci_upper=float(inf["hi"][i]),
            )
            for i, nm in enumerate(pnames)
        }

        return {
            "model":        "OLS",
            "se_type":      "HC1" if self.robust else "OLS",
            "n_obs":        n,
            "n_features":   k_reg,
            "df_model":     k_reg,
            "df_resid":     n - k,
            "coefficients": coef_tbl,
            "fit": {
                "R2":     r2,
                "adj_R2": adj_r2,
                "F_stat": f_stat,
                "F_pval": f_pval,
                "sigma":  sigma,
                "MSE":    metrics["MSE"],
                "RMSE":   metrics["RMSE"],
                "MAE":    metrics["MAE"],
            },
            "residuals": {
                "mean": float(self._residuals.mean()),
                "std":  float(self._residuals.std(ddof=k)),
                "min":  float(self._residuals.min()),
                "max":  float(self._residuals.max()),
            },
        }

    def summary_table(self) -> pd.DataFrame:
        """以 DataFrame 格式返回系数推断表，方便 notebook 展示。

        返回
        ----
        pd.DataFrame，index=参数名，列=coef/se/t/p_value/ci_lower/ci_upper
        """
        s = self.summary()
        rows = [{"parameter": nm, **vals} for nm, vals in s["coefficients"].items()]
        return pd.DataFrame(rows).set_index("parameter")


# ── 辅助数据生成 ──────────────────────────────────────────────────────

def mincer_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成 Mincer (1974) 工资方程合成数据集。

    真实 DGP：
        ln_wage = 1.2 + 0.08*educ + 0.04*exper - 0.0007*exper^2 + eps
        eps ~ N(0, 0.35^2)

    参数
    ----
    n    : 样本量，默认 500
    seed : 随机种子，默认 42

    返回
    ----
    pd.DataFrame，列：educ / exper / exper2 / ln_wage / wage
    """
    rng = np.random.default_rng(seed)
    educ   = rng.integers(6, 20, size=n).astype(float)
    exper  = rng.integers(0, 40, size=n).astype(float)
    exper2 = exper ** 2
    eps    = rng.normal(0, 0.35, size=n)
    ln_wage = 1.2 + 0.08 * educ + 0.04 * exper - 0.0007 * exper2 + eps
    return pd.DataFrame(dict(
        educ=educ, exper=exper, exper2=exper2,
        ln_wage=ln_wage, wage=np.exp(ln_wage),
    ))


def demo_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成通用合成数据（向后兼容保留）。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y  = 2.0 + 1.6*x1 - 0.9*x2 + 0.4*x3 + rng.normal(scale=0.8, size=n)
    return pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, y=y))


# ── 最小可运行示例 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("示例 1：通用合成数据（经典 OLS 标准误）")
    print("=" * 60)
    df1 = demo_data()
    m1 = OLS(robust=False).fit(df1[["x1","x2","x3"]], df1["y"])
    pprint.pprint(m1.summary())
    print()
    print(m1.summary_table().round(4))

    print()
    print("=" * 60)
    print("示例 2：Mincer 工资方程（HC1 稳健标准误）")
    print("=" * 60)
    df2 = mincer_data(n=500)
    m2 = OLS(robust=True).fit(df2[["educ","exper","exper2"]], df2["ln_wage"])
    pprint.pprint(m2.summary())
    print()
    print(m2.summary_table().round(4))
