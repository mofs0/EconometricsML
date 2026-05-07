"""两阶段最小二乘回归 (IV / 2SLS)
====================================
来源参考：Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data, Ch.5;
          Acemoglu, Johnson & Robinson (2001). "The Colonial Origins of Comparative Development". AER.
适用场景：处理自变量内生性，使用工具变量进行一致估计。
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


class IV2SLS:
    """两阶段最小二乘 (Two-Stage Least Squares).

    第一阶段：对内生变量 D 回归工具变量 Z 与外生变量 X。
    第二阶段：用拟合值 D_hat 替换 D，再做 OLS。
    标准误采用 2SLS 渐近方差（默认）或 HC1 稳健形式。

    参数
    ----
    fit_intercept : bool, default True
    robust        : bool, default False  HC1 稳健标准误

    示例
    ----
    >>> from empirlab.traditional.iv import IV2SLS, ajr_data
    >>> df = ajr_data()
    >>> m = IV2SLS(robust=True)
    >>> m.fit(df[['risk']], df['loggdp'], df[['logmort']], df[['lat_abst','africa','asia']])
    >>> print(m.summary_table().round(4))
    """

    def __init__(self, fit_intercept: bool = True, robust: bool = False):
        self.fit_intercept = fit_intercept
        self.robust = robust
        self.is_fitted = False
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self._first_stage_r2: Optional[float] = None
        self._first_stage_f: Optional[float] = None
        self._X_endog: Optional[np.ndarray] = None
        self._Z: Optional[np.ndarray] = None
        self._X_design2: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._beta_full: Optional[np.ndarray] = None

    def _to_numpy(self, arr, store_names=False):
        if isinstance(arr, pd.DataFrame):
            if store_names:
                self.feature_names_ = list(arr.columns)
            return arr.to_numpy(dtype=float)
        if isinstance(arr, pd.Series):
            return arr.to_numpy(dtype=float)
        return np.asarray(arr, dtype=float)

    def _ols_fit(self, X, y):
        return np.linalg.pinv(X) @ y

    def _se(self, X2, resid, n, k):
        rss = float(np.sum(resid**2))
        if self.robust:
            XtX_inv = np.linalg.pinv(X2.T @ X2)
            Xe = X2 * resid[:, None]
            meat = Xe.T @ Xe
            cov = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
        else:
            sigma2 = rss / (n - k)
            cov = sigma2 * np.linalg.pinv(X2.T @ X2)
        return np.sqrt(np.diag(cov))

    def fit(self, X_endog, y, Z, X_exog=None) -> "IV2SLS":
        """拟合 2SLS 模型。

        参数
        ----
        X_endog : 内生自变量矩阵，shape (n, p_endog)
        y       : 因变量，shape (n,)
        Z       : 工具变量矩阵，shape (n, p_z)，要求 p_z >= p_endog
        X_exog  : 外生控制变量，shape (n, p_exog)，可为 None
        """
        X_end = self._to_numpy(X_endog, store_names=True)
        y_arr = self._to_numpy(y).ravel()
        Z_arr = self._to_numpy(Z)
        X_ex  = self._to_numpy(X_exog) if X_exog is not None else np.empty((len(y_arr), 0))

        n = len(y_arr)
        ones = np.ones((n, 1))

        # 第一阶段：D ~ Z + X_exog + const
        Z_full = np.hstack([ones, Z_arr, X_ex]) if self.fit_intercept else np.hstack([Z_arr, X_ex])
        beta1 = self._ols_fit(Z_full, X_end)
        X_end_hat = Z_full @ beta1
        # 保存第一阶段 R² 和 F 统计量（识别强度检验）
        y1 = X_end[:, 0]
        y1_hat = X_end_hat[:, 0]
        ss_res = np.sum((y1 - y1_hat)**2)
        ss_tot = np.sum((y1 - y1.mean())**2)
        self._first_stage_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        k1 = Z_full.shape[1]
        f1 = (self._first_stage_r2 / (k1 - 1)) / ((1 - self._first_stage_r2) / (n - k1))
        self._first_stage_f = float(f1)

        # 第二阶段：y ~ D_hat + X_exog + const
        X2 = np.hstack([ones, X_end_hat, X_ex]) if self.fit_intercept else np.hstack([X_end_hat, X_ex])
        self._beta_full = self._ols_fit(X2, y_arr)
        self._X_endog = X_end
        self._Z = Z_arr
        self._X_design2 = X2
        self._y = y_arr
        self._residuals = y_arr - X2 @ self._beta_full

        if self.fit_intercept:
            self.coef_ = self._beta_full[1:]
        else:
            self.coef_ = self._beta_full
        self.is_fitted = True
        return self

    def predict(self, X_endog, X_exog=None) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()。")
        X_end = np.asarray(X_endog, dtype=float)
        X_ex  = np.asarray(X_exog, dtype=float) if X_exog is not None else np.empty((len(X_end), 0))
        n = len(X_end)
        ones = np.ones((n, 1))
        X2 = np.hstack([ones, X_end, X_ex]) if self.fit_intercept else np.hstack([X_end, X_ex])
        return X2 @ self._beta_full

    def summary(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        n, k = self._X_design2.shape
        k_reg = k - (1 if self.fit_intercept else 0)
        se = self._se(self._X_design2, self._residuals, n, k)
        t = self._beta_full / se
        df = n - k
        p = 2 * stats.t.sf(np.abs(t), df=df)
        cv = stats.t.ppf(0.975, df=df)
        metrics = calculate_metrics(self._y, self._X_design2 @ self._beta_full)

        if self.feature_names_ is not None:
            names = self.feature_names_
        else:
            names = [f"endog{i+1}" for i in range(self._X_endog.shape[1])]
            if self._X_design2.shape[1] > 1 + self._X_endog.shape[1]:
                names += [f"exog{i+1}" for i in range(self._X_design2.shape[1] - 1 - self._X_endog.shape[1])]
        pnames = (["const"] + names) if self.fit_intercept else names

        coef_tbl = {
            nm: dict(coef=float(self._beta_full[i]), se=float(se[i]),
                     t=float(t[i]), p_value=float(p[i]),
                     ci_lower=float(self._beta_full[i] - cv * se[i]),
                     ci_upper=float(self._beta_full[i] + cv * se[i]))
            for i, nm in enumerate(pnames)
        }
        return {
            "model": "IV-2SLS",
            "se_type": "HC1" if self.robust else "OLS",
            "n_obs": n, "n_features": k_reg, "df_resid": df,
            "coefficients": coef_tbl,
            "first_stage": {
                "R2": self._first_stage_r2,
                "F_stat": self._first_stage_f,
                "note": "F > 10 通常认为工具变量不弱（Staiger & Stock 1997）",
            },
            "fit": {"R2": metrics["R2"], "RMSE": metrics["RMSE"]},
        }

    def summary_table(self) -> pd.DataFrame:
        s = self.summary()
        rows = [{"parameter": nm, **vals} for nm, vals in s["coefficients"].items()]
        return pd.DataFrame(rows).set_index("parameter")


def ajr_data(n: int = 64, seed: int = 42) -> pd.DataFrame:
    """AJR (2001) 风格合成数据：定居者死亡率作为制度质量的 IV。

    DGP（基于 AJR 2001 AER）：
        loggdp  = 1.0 + 0.9*risk + 0.5*lat + eps_y
        risk    = 4.5 - 0.8*logmort + 0.3*lat + eps_d   (内生)
        logmort = DGP 外生（工具变量）
    """
    rng = np.random.default_rng(seed)
    logmort  = rng.normal(4.7, 1.2, n)
    lat_abst = rng.uniform(0, 0.7, n)
    africa   = (rng.uniform(size=n) < 0.3).astype(float)
    asia     = (rng.uniform(size=n) < 0.25).astype(float)

    eps_d = rng.normal(0, 0.6, n)
    risk  = 4.5 - 0.8 * logmort + 0.3 * lat_abst + eps_d

    eps_y = rng.normal(0, 0.4, n) + 0.3 * eps_d  # 内生性来源
    loggdp = 1.0 + 0.9 * risk + 0.5 * lat_abst + eps_y

    return pd.DataFrame(dict(loggdp=loggdp, risk=risk, logmort=logmort,
                             lat_abst=lat_abst, africa=africa, asia=asia))


if __name__ == "__main__":
    import pprint
    df = ajr_data()
    m = IV2SLS(robust=True)
    m.fit(df[['risk']], df['loggdp'], df[['logmort']], df[['lat_abst','africa','asia']])
    pprint.pprint(m.summary())
    print(); print(m.summary_table().round(4))
