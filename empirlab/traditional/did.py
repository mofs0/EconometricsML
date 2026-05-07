"""双重差分法 (Difference-in-Differences)
==========================================
来源参考：Angrist & Pischke (2009). Mostly Harmless Econometrics, Ch.5;
          Fang, T. & Lin, C. (2015). "Minimum Wages and Employment in China".
          IZA Journal of Labor Policy, 4(22). DOI:10.1186/s40173-015-0050-9
适用场景：政策评估，利用政策时间与处理组/控制组的双重变动识别因果效应。
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


class DiD:
    """双重差分估计量 (Difference-in-Differences / 2x2 DiD).

    标准 DiD 方程：
        y_it = α + β₁·Treat_i + β₂·Post_t + δ·(Treat_i × Post_t) + X_it·γ + ε_it

    δ 即政策 ATT（平均处理效应）。

    参数
    ----
    robust : bool, default True  HC1 稳健标准误

    示例
    ----
    >>> from empirlab.traditional.did import DiD, minwage_data
    >>> df = minwage_data()
    >>> m = DiD(robust=True)
    >>> m.fit(df, treat_col='treat', post_col='post', y_col='lnemp',
    ...       covariates=['lngdp','lnpop'])
    >>> print(m.summary_table().round(4))
    """

    def __init__(self, robust: bool = True):
        self.robust = robust
        self.is_fitted = False
        self.coef_: Optional[np.ndarray] = None
        self._X_design: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._resid: Optional[np.ndarray] = None
        self._param_names: Optional[List[str]] = None

    def _ols(self, X, y):
        return np.linalg.pinv(X) @ y

    def _se(self, X, resid):
        n, k = X.shape
        df = n - k
        if self.robust:
            XtX_inv = np.linalg.pinv(X.T @ X)
            Xe = X * resid[:, None]
            meat = Xe.T @ Xe
            cov = (n / df) * XtX_inv @ meat @ XtX_inv
        else:
            sigma2 = np.sum(resid**2) / df
            cov = sigma2 * np.linalg.pinv(X.T @ X)
        return np.sqrt(np.diag(cov))

    def fit(self, df: pd.DataFrame,
            treat_col: str, post_col: str, y_col: str,
            covariates: Optional[List[str]] = None,
            entity_col: Optional[str] = None) -> "DiD":
        """拟合 DiD 模型。

        参数
        ----
        df          : 长格式面板 DataFrame
        treat_col   : 处理组虚拟变量列（0/1）
        post_col    : 政策后虚拟变量列（0/1）
        y_col       : 因变量列
        covariates  : 额外控制变量列表
        entity_col  : 若提供，使用个体固定效应（within 变换）
        """
        df2 = df.copy()
        df2['_did'] = df2[treat_col] * df2[post_col]

        cols = [treat_col, post_col, '_did'] + (covariates or [])
        X_raw = df2[cols].to_numpy(dtype=float)
        X = np.hstack([np.ones((len(df2), 1)), X_raw])
        y = df2[y_col].to_numpy(dtype=float)

        self._param_names = ['const', treat_col, post_col, 'treat×post'] + (covariates or [])
        self._X_design = X
        self._y = y
        self.coef_ = self._ols(X, y)
        self._resid = y - X @ self.coef_
        self.is_fitted = True
        return self

    def summary(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        n, k = self._X_design.shape
        se = self._se(self._X_design, self._resid)
        t = self.coef_ / se
        df = n - k
        p = 2 * stats.t.sf(np.abs(t), df=df)
        cv = stats.t.ppf(0.975, df=df)
        r2 = float(1 - np.sum(self._resid**2) / np.sum((self._y - self._y.mean())**2))
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if n > k + 1 else np.nan
        coef_tbl = {
            nm: dict(coef=float(self.coef_[i]), se=float(se[i]),
                     t=float(t[i]), p_value=float(p[i]),
                     ci_lower=float(self.coef_[i] - cv * se[i]),
                     ci_upper=float(self.coef_[i] + cv * se[i]))
            for i, nm in enumerate(self._param_names)
        }
        att = self.coef_[3]  # treat×post 系数
        return {
            "model": "DiD",
            "se_type": "HC1" if self.robust else "OLS",
            "n_obs": n,
            "ATT": float(att),
            "ATT_pct": float((np.exp(att) - 1) * 100) if abs(att) < 2 else np.nan,
            "coefficients": coef_tbl,
            "fit": {"R2": r2, "adj_R2": adj_r2},
            "note": "ATT_pct: 若 y 为对数，ATT 对应的百分比效应。"
        }

    def summary_table(self) -> pd.DataFrame:
        s = self.summary()
        rows = [{"parameter": nm, **vals} for nm, vals in s["coefficients"].items()]
        return pd.DataFrame(rows).set_index("parameter")

    def parallel_trends_plot_data(self, df: pd.DataFrame,
                                  treat_col: str, time_col: str,
                                  y_col: str) -> pd.DataFrame:
        """返回平行趋势图所需的分组均值数据。"""
        return (df.groupby([time_col, treat_col])[y_col]
                  .mean().reset_index()
                  .rename(columns={treat_col: 'group', y_col: 'mean_y'}))


def minwage_data(n_counties: int = 100, n_years: int = 6,
                 treat_frac: float = 0.4, seed: int = 42) -> pd.DataFrame:
    """Fang & Lin (2015) 风格：最低工资政策对就业影响的合成面板数据。

    政策：第 3 年起处理组县/市提高最低工资标准。
    DGP：lnemp = 2.0 + (-0.05)*treat×post + 0.3*lngdp + 0.2*lnpop + alpha_i + eps
    """
    rng = np.random.default_rng(seed)
    county_ids = np.repeat(np.arange(1, n_counties + 1), n_years)
    years = np.tile(np.arange(2015, 2015 + n_years), n_counties)
    treat_arr = (np.arange(1, n_counties + 1) <= int(n_counties * treat_frac)).astype(float)
    treat = np.repeat(treat_arr, n_years)
    post  = (years >= 2018).astype(float)

    alpha_i = rng.normal(0, 0.3, n_counties)
    alpha   = np.repeat(alpha_i, n_years)
    lngdp   = rng.normal(8.0, 0.5, n_counties * n_years)
    lnpop   = rng.normal(5.5, 0.4, n_counties * n_years)
    eps     = rng.normal(0, 0.2, n_counties * n_years)

    lnemp = 2.0 + (-0.05) * treat * post + 0.3 * lngdp + 0.2 * lnpop + alpha + eps
    return pd.DataFrame(dict(county_id=county_ids, year=years, treat=treat, post=post,
                             lnemp=lnemp, lngdp=lngdp, lnpop=lnpop))


if __name__ == "__main__":
    import pprint
    df = minwage_data()
    m = DiD(robust=True)
    m.fit(df, 'treat', 'post', 'lnemp', covariates=['lngdp','lnpop'])
    print("ATT =", round(m.summary()['ATT'], 4),
          "  (百分比效应:", round(m.summary()['ATT_pct'], 2), "%)")
    print(); print(m.summary_table().round(4))
