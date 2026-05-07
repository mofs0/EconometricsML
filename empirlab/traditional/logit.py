"""Logit / Probit 二元离散选择模型
====================================
来源参考：Greene, W. H. (2003). Econometric Analysis, Ch.21;
          Roberts & Tybout (1997). "The Decision to Export in Colombia". AER.
适用场景：二元因变量（0/1）的概率建模，含边际效应与模型诊断。
Python 版本：3.10+
依赖：numpy >= 1.24, pandas >= 1.5, scipy >= 1.10
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from empirlab.utils.metrics import calculate_metrics


def _logistic(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return stats.norm.cdf(z)

def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return stats.norm.pdf(z)


class Logit:
    """Logit / Probit 二元离散选择模型 (MLE).

    使用最大似然估计，支持 logit（逻辑斯谛）和 probit（正态）两种连接函数。

    参数
    ----
    model : {'logit', 'probit'}, default 'logit'
        连接函数类型。
    fit_intercept : bool, default True
        是否加入截距项。

    属性
    ----
    coef_          : np.ndarray  完整系数向量（含截距）
    feature_names_ : list[str] or None
    is_fitted      : bool

    示例
    ----
    >>> import numpy as np
    >>> from empirlab.traditional.logit import Logit, export_data
    >>> df = export_data(n=500)
    >>> m = Logit(model='logit').fit(df[['lnsize','lnprod','fdi']], df['export'])
    >>> print(m.summary_table().round(4))
    """

    def __init__(self,
                 model: Literal['logit', 'probit'] = 'logit',
                 fit_intercept: bool = True):
        self.model = model
        self.fit_intercept = fit_intercept
        self.is_fitted = False
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self._X_design: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._n_iter: int = 0

    # ── 内部 ────────────────────────────────────────────────────────────

    def _to_numpy(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _link(self, z):
        return _logistic(z) if self.model == 'logit' else _normal_cdf(z)

    def _link_deriv(self, z):
        if self.model == 'logit':
            p = _logistic(z)
            return p * (1 - p)
        return _normal_pdf(z)

    def _neg_loglik(self, beta, X, y):
        z = X @ beta
        p = np.clip(self._link(z), 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _neg_loglik_grad(self, beta, X, y):
        z = X @ beta
        p = self._link(z)
        resid = y - p
        return -X.T @ resid

    def _hessian(self):
        z = self._X_design @ self.coef_
        w = self._link_deriv(z)
        return self._X_design.T @ (self._X_design * w[:, None])

    def _std_errors(self) -> np.ndarray:
        H = self._hessian()
        cov = np.linalg.pinv(H)
        return np.sqrt(np.diag(cov))

    # ── 公开接口 ────────────────────────────────────────────────────────

    def fit(self, X, y) -> "Logit":
        """MLE 拟合。

        参数
        ----
        X : array-like or DataFrame, shape (n, p)
        y : array-like, shape (n,)  取值 0 或 1
        """
        X_arr = self._to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        if X_arr.ndim != 2:
            raise ValueError("X 必须是二维矩阵。")
        if not np.all(np.isin(y_arr, [0.0, 1.0])):
            raise ValueError("y 必须是 0/1 二元变量。")

        ones = np.ones((X_arr.shape[0], 1))
        self._X_design = np.hstack([ones, X_arr]) if self.fit_intercept else X_arr
        self._y = y_arr

        beta0 = np.zeros(self._X_design.shape[1])
        result = optimize.minimize(
            self._neg_loglik, beta0,
            jac=self._neg_loglik_grad,
            args=(self._X_design, y_arr),
            method='BFGS',
            options={'maxiter': 1000, 'gtol': 1e-6}
        )
        self.coef_ = result.x
        self._n_iter = result.nit
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """返回 P(y=1|X) 的预测概率。"""
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()。")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        ones = np.ones((X_arr.shape[0], 1))
        X_d = np.hstack([ones, X_arr]) if self.fit_intercept else X_arr
        return self._link(X_d @ self.coef_)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """返回 0/1 预测类别（默认阈值 0.5）。"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def marginal_effects(self) -> np.ndarray:
        """计算平均边际效应 (AME)。

        AME_j = (1/n) * sum_i [ dP/dX_j |_{X=x_i} ]
        """
        z = self._X_design @ self.coef_
        dydz = self._link_deriv(z).mean()
        k_reg = len(self.coef_) - (1 if self.fit_intercept else 0)
        coef_x = self.coef_[1:] if self.fit_intercept else self.coef_
        return dydz * coef_x

    def summary(self) -> Dict[str, object]:
        """返回完整 MLE 摘要字典。"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")

        n, k = self._X_design.shape
        k_reg = k - (1 if self.fit_intercept else 0)
        se = self._std_errors()
        z_stat = self.coef_ / se
        p_val = 2 * stats.norm.sf(np.abs(z_stat))
        cv = stats.norm.ppf(0.975)

        names = self.feature_names_ or [f"x{i+1}" for i in range(k_reg)]
        pnames = (["const"] + names) if self.fit_intercept else names

        coef_tbl = {
            nm: dict(
                coef=float(self.coef_[i]),
                se=float(se[i]),
                z=float(z_stat[i]),
                p_value=float(p_val[i]),
                ci_lower=float(self.coef_[i] - cv * se[i]),
                ci_upper=float(self.coef_[i] + cv * se[i]),
            )
            for i, nm in enumerate(pnames)
        }

        # 对数似然与伪 R²
        z = self._X_design @ self.coef_
        p = np.clip(self._link(z), 1e-15, 1 - 1e-15)
        ll_full = float(np.sum(self._y * np.log(p) + (1 - self._y) * np.log(1 - p)))
        p0 = self._y.mean()
        ll_null = float(n * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))
        mcfadden_r2 = 1 - ll_full / ll_null

        y_hat_class = (p >= 0.5).astype(int)
        accuracy = float(np.mean(y_hat_class == self._y))

        ame = self.marginal_effects()
        ame_dict = {nm: float(ame[i]) for i, nm in enumerate(names)}

        return {
            "model":           self.model.capitalize(),
            "n_obs":           n,
            "n_features":      k_reg,
            "df_resid":        n - k,
            "coefficients":    coef_tbl,
            "marginal_effects": ame_dict,
            "fit": {
                "log_likelihood": ll_full,
                "McFadden_R2":   mcfadden_r2,
                "AIC":           -2 * ll_full + 2 * k,
                "BIC":           -2 * ll_full + k * np.log(n),
                "accuracy":      accuracy,
            },
            "n_iter": self._n_iter,
        }

    def summary_table(self) -> pd.DataFrame:
        """返回系数推断 DataFrame（含边际效应列）。"""
        s = self.summary()
        rows = [{"parameter": nm, **vals} for nm, vals in s["coefficients"].items()]
        df = pd.DataFrame(rows).set_index("parameter")
        ame = s["marginal_effects"]
        df["AME"] = [ame.get(p, float("nan")) for p in df.index]
        return df


# ── 数据生成 ────────────────────────────────────────────────────────────

def export_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Roberts & Tybout (1997) 风格：企业出口决策合成数据。

    DGP：export* = β0 + β1*lnsize + β2*lnprod + β3*fdi + ε
         export  = 1{export* > 0}
    真实参数：β = [-3.0, 0.6, 0.8, 1.2]
    """
    rng = np.random.default_rng(seed)
    lnsize = rng.normal(4.5, 1.2, n)
    lnprod = rng.normal(3.0, 0.8, n)
    fdi    = rng.binomial(1, 0.15, n).astype(float)
    xb     = -3.0 + 0.6 * lnsize + 0.8 * lnprod + 1.2 * fdi
    export = (xb + rng.logistic(size=n) > 0).astype(float)
    return pd.DataFrame(dict(lnsize=lnsize, lnprod=lnprod, fdi=fdi, export=export))


# ── 最小可运行示例 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    df = export_data(n=1000)
    print(f"出口比例：{df['export'].mean():.3f}")

    for mtype in ['logit', 'probit']:
        print(f"\n{'='*55}")
        print(f"  {mtype.upper()} 模型")
        print('='*55)
        m = Logit(model=mtype).fit(df[['lnsize','lnprod','fdi']], df['export'])
        pprint.pprint(m.summary())
        print()
        print(m.summary_table().round(4))
