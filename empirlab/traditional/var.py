"""向量自回归模型 (VAR)
========================
来源参考：Sims (1980). "Macroeconomics and Reality". Econometrica;
          陈六傅, 刘厚俊 (2008). "人民币汇率的价格传递效应——基于VAR模型的实证分析".
          《经济研究》第7期.
适用场景：多变量时序系统的动态分析：脉冲响应、方差分解、格兰杰因果检验。
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


class VAR:
    """向量自回归模型 VAR(p).

    对 k 维时序向量 y_t 估计：
        y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t

    参数
    ----
    lags : int, default 2  滞后阶数 p

    示例
    ----
    >>> from empirlab.traditional.var import VAR, macro_data
    >>> df = macro_data()
    >>> m = VAR(lags=2).fit(df[['lnm2','lncpi','lnrgdp','r']])
    >>> irf = m.irf(h=10)
    >>> print(m.summary_table())
    """

    def __init__(self, lags: int = 2):
        self.lags = lags
        self.is_fitted = False
        self._coef: Optional[np.ndarray] = None
        self._sigma_u: Optional[np.ndarray] = None
        self._var_names: Optional[List[str]] = None
        self._y: Optional[np.ndarray] = None
        self._resid: Optional[np.ndarray] = None
        self._n: int = 0
        self._k: int = 0

    def _build_lagged(self, Y: np.ndarray, p: int):
        """构建 VAR 设计矩阵 [1, y_{t-1}, ..., y_{t-p}]。"""
        T, k = Y.shape
        rows = []
        for t in range(p, T):
            row = [1.0]
            for lag in range(1, p+1):
                row.extend(Y[t-lag])
            rows.append(row)
        X = np.array(rows)
        Y_dep = Y[p:]
        return X, Y_dep

    def fit(self, data) -> "VAR":
        """拟合 VAR(p) 模型（方程逐列 OLS）。

        参数
        ----
        data : pd.DataFrame 或 np.ndarray，列为变量，行为时间
        """
        if isinstance(data, pd.DataFrame):
            self._var_names = list(data.columns)
            Y = data.to_numpy(dtype=float)
        else:
            Y = np.asarray(data, dtype=float)
            self._var_names = [f"y{i+1}" for i in range(Y.shape[1])]

        T, k = Y.shape
        self._k = k
        X, Y_dep = self._build_lagged(Y, self.lags)
        self._n = len(Y_dep)
        n_params = 1 + k * self.lags  # const + k*p coefs per equation

        # 系数矩阵：shape (n_params, k)
        self._coef = np.linalg.pinv(X) @ Y_dep
        self._resid = Y_dep - X @ self._coef
        self._sigma_u = (self._resid.T @ self._resid) / (self._n - n_params)
        self._X_design = X
        self._Y_dep = Y_dep
        self._Y_full = Y
        self.is_fitted = True
        return self

    def _companion_matrix(self) -> np.ndarray:
        """构建伴随矩阵（用于 IRF 计算）。"""
        k, p = self._k, self.lags
        A = np.zeros((k*p, k*p))
        for lag in range(p):
            A[:k, lag*k:(lag+1)*k] = self._coef[1+lag*k:1+(lag+1)*k, :].T
        if p > 1:
            A[k:, :k*(p-1)] = np.eye(k*(p-1))
        return A

    def irf(self, h: int = 10, orthogonalize: bool = True) -> np.ndarray:
        """计算脉冲响应函数 (IRF).

        参数
        ----
        h              : 预测步数
        orthogonalize  : 是否用 Cholesky 正交化（OIRF）

        返回
        ----
        np.ndarray, shape (h+1, k, k)  [period, response, shock]
        """
        k, p = self._k, self.lags
        A = self._companion_matrix()
        if orthogonalize:
            try:
                P = np.linalg.cholesky(self._sigma_u)
            except np.linalg.LinAlgError:
                P = np.eye(k)
        else:
            P = np.eye(k)

        Phi = np.zeros((h+1, k*p, k*p))
        Phi[0] = np.eye(k*p)
        for i in range(1, h+1):
            Phi[i] = Phi[i-1] @ A.T

        irf_mat = np.zeros((h+1, k, k))
        for i in range(h+1):
            irf_mat[i] = Phi[i][:k, :k] @ P
        return irf_mat

    def fevd(self, h: int = 10) -> np.ndarray:
        """预测误差方差分解 (FEVD).

        返回
        ----
        np.ndarray, shape (h, k, k)  [period, variable, shock_source]
        """
        irf = self.irf(h=h, orthogonalize=True)
        mse = np.cumsum(irf**2, axis=0)
        total = mse.sum(axis=2, keepdims=True)
        return mse[1:] / np.where(total[1:] > 0, total[1:], 1.0)

    def granger_causality(self, cause_var: str, effect_var: str) -> dict:
        """格兰杰因果检验：cause_var 是否格兰杰导致 effect_var。"""
        names = self._var_names
        j = names.index(effect_var)
        k, p = self._k, self.lags
        # 受限模型：移除 cause_var 的所有滞后项
        i_cause = names.index(cause_var)
        restrict_cols = [1 + lag*k + i_cause for lag in range(p)]

        X_ur = self._X_design
        X_r  = np.delete(X_ur, restrict_cols, axis=1)
        y_j  = self._Y_dep[:, j]

        beta_ur = np.linalg.pinv(X_ur) @ y_j
        beta_r  = np.linalg.pinv(X_r) @ y_j
        rss_ur = float(np.sum((y_j - X_ur @ beta_ur)**2))
        rss_r  = float(np.sum((y_j - X_r @ beta_r)**2))
        df1 = p
        df2 = self._n - X_ur.shape[1]
        f_stat = ((rss_r - rss_ur) / df1) / (rss_ur / df2)
        p_val  = float(stats.f.sf(f_stat, df1, df2))
        return {
            "cause": cause_var, "effect": effect_var,
            "F_stat": float(f_stat), "p_value": p_val,
            "conclusion": f"{cause_var} {'格兰杰导致' if p_val < 0.05 else '不格兰杰导致'} {effect_var}"
        }

    def summary(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        k, p = self._k, self.lags
        n_params = 1 + k * p
        aic = np.log(np.linalg.det(self._sigma_u)) + 2 * k * n_params / self._n
        bic = np.log(np.linalg.det(self._sigma_u)) + np.log(self._n) * k * n_params / self._n
        return {
            "model": f"VAR({self.lags})",
            "n_obs": self._n,
            "n_vars": k,
            "var_names": self._var_names,
            "lags": self.lags,
            "AIC": float(aic),
            "BIC": float(bic),
            "sigma_u_diag": self._sigma_u.diagonal().tolist(),
        }

    def summary_table(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        k, p = self._k, self.lags
        pnames = ['const'] + [f'{v}_L{l}' for l in range(1,p+1) for v in self._var_names]
        rows = []
        for j, yname in enumerate(self._var_names):
            for i, pname in enumerate(pnames):
                rows.append({'equation': yname, 'parameter': pname,
                             'coef': float(self._coef[i, j])})
        return pd.DataFrame(rows).set_index(['equation','parameter'])


def macro_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """陈六傅 & 刘厚俊 (2008) 风格：中国宏观季度数据合成。

    变量：lnm2（货币供给）, lncpi（物价）, lnrgdp（实际产出）, r（利率）
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    lnrgdp = 8.0 + 0.015*t + rng.normal(0, 0.05, n)
    lnm2   = 7.0 + 0.018*t + 0.4*lnrgdp[-n:] + rng.normal(0, 0.06, n)
    lncpi  = 4.5 + 0.008*t + 0.3*lnm2 + rng.normal(0, 0.04, n)
    r      = 0.04 + 0.3*(lncpi - lncpi.mean()) + rng.normal(0, 0.005, n)
    quarters = pd.date_range('2000Q1', periods=n, freq='QS')
    return pd.DataFrame(dict(date=quarters, lnm2=lnm2, lncpi=lncpi, lnrgdp=lnrgdp, r=r))


if __name__ == "__main__":
    import pprint
    df = macro_data()
    m = VAR(lags=2).fit(df[['lnm2','lncpi','lnrgdp','r']])
    pprint.pprint(m.summary())
    print(); print(m.summary_table().head(12).round(4))
    gc = m.granger_causality('lnm2', 'lncpi')
    print(f"\n格兰杰因果：{gc['conclusion']}  (F={gc['F_stat']:.3f}, p={gc['p_value']:.4f})")
