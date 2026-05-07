"""GARCH 族波动率模型
======================
来源参考：Bollerslev (1986). "Generalized Autoregressive Conditional Heteroskedasticity". JoE;
          方意, 荣雪 (2019). "金融风险跨市场传染的宏观审慎监管研究".
          《管理世界》第35卷第2期.
适用场景：金融时序波动率建模，捕捉波动聚集性（volatility clustering）。
Python 版本：3.10+
依赖：numpy >= 1.24, pandas >= 1.5, scipy >= 1.10
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import optimize, stats

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class GARCH:
    """GARCH(p, q) 模型 (Bollerslev 1986).

    均值方程：r_t = mu + eps_t，eps_t = sigma_t * z_t，z_t ~ N(0,1)
    方差方程：sigma_t^2 = omega + sum_i(alpha_i * eps_{t-i}^2) + sum_j(beta_j * sigma_{t-j}^2)

    参数
    ----
    p : int, default 1  ARCH 项数（滞后 eps^2）
    q : int, default 1  GARCH 项数（滞后 sigma^2）

    示例
    ----
    >>> from empirlab.traditional.garch import GARCH, stock_return_data
    >>> r = stock_return_data()
    >>> m = GARCH(p=1, q=1).fit(r['ret'])
    >>> print(m.summary_table().round(6))
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.is_fitted = False
        self._params: Optional[np.ndarray] = None
        self._sigma2: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._loglik: Optional[float] = None

    def _log_likelihood(self, params, eps):
        """高斯 GARCH 对数似然（负值用于最小化）。"""
        n = len(eps)
        mu   = params[0]
        omega = params[1]
        alpha = params[2:2+self.p]
        beta  = params[2+self.p:2+self.p+self.q]

        e = eps - mu
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(e)

        for t in range(1, n):
            s2 = omega
            for i in range(self.p):
                if t - i - 1 >= 0:
                    s2 += alpha[i] * e[t-i-1]**2
            for j in range(self.q):
                if t - j - 1 >= 0:
                    s2 += beta[j] * sigma2[t-j-1]
            sigma2[t] = max(s2, 1e-8)

        ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(sigma2) + e**2 / sigma2)
        return -ll, sigma2, e

    def _neg_ll_scalar(self, params, eps):
        return self._log_likelihood(params, eps)[0]

    def fit(self, returns) -> "GARCH":
        """MLE 拟合 GARCH(p,q) 模型。

        参数
        ----
        returns : array-like  收益率序列（均值为零附近的对数收益）
        """
        r = np.asarray(returns, dtype=float)
        n_params = 2 + self.p + self.q  # mu, omega, alpha_1..p, beta_1..q
        x0 = np.array([np.mean(r), np.var(r) * 0.05] +
                      [0.1] * self.p + [0.8] * self.q)

        # 约束：omega > 0，alpha >= 0，beta >= 0，sum(alpha+beta) < 1
        bounds = [(None, None), (1e-8, None)] + \
                 [(1e-8, 0.999)] * self.p + [(1e-8, 0.999)] * self.q
        constraints = [{'type': 'ineq',
                        'fun': lambda x: 0.999 - np.sum(x[2+self.p:])}]
        result = optimize.minimize(
            self._neg_ll_scalar, x0, args=(r,),
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-8}
        )
        self._params = result.x
        neg_ll, sigma2, e = self._log_likelihood(result.x, r)
        self._loglik = -float(neg_ll)
        self._sigma2 = sigma2
        self._residuals = e
        self._n = len(r)
        self.is_fitted = True
        return self

    def _hessian_se(self, eps):
        """数值 Hessian 估计标准误。"""
        from scipy.optimize import approx_fprime
        h = 1e-5
        n_p = len(self._params)
        H = np.zeros((n_p, n_p))
        for i in range(n_p):
            def f_i(x):
                p2 = self._params.copy()
                p2[i] = x
                return self._neg_ll_scalar(p2, eps)
            grad_plus  = approx_fprime([self._params[i] + h], f_i, h)[0]
            grad_minus = approx_fprime([self._params[i] - h], f_i, h)[0]
            H[i, i] = (grad_plus - grad_minus) / (2 * h)
        cov = np.linalg.pinv(H + np.eye(n_p)*1e-6)
        return np.sqrt(np.maximum(np.diag(cov), 0))

    def predict_variance(self, h: int = 1) -> np.ndarray:
        """h 步前向预测条件方差。"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        sigma2_fwd = np.zeros(h)
        e = self._residuals
        s2 = self._sigma2
        mu, omega = self._params[0], self._params[1]
        alpha = self._params[2:2+self.p]
        beta  = self._params[2+self.p:2+self.p+self.q]
        for t in range(h):
            s = omega
            for i in range(self.p):
                if t - i - 1 >= 0:
                    s += alpha[i] * (mu**2 + sigma2_fwd[t-i-1])
                else:
                    s += alpha[i] * e[-(i+1)]**2
            for j in range(self.q):
                if t - j - 1 >= 0:
                    s += beta[j] * sigma2_fwd[t-j-1]
                else:
                    s += beta[j] * s2[-(j+1)]
            sigma2_fwd[t] = s
        return sigma2_fwd

    def summary(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        mu, omega = float(self._params[0]), float(self._params[1])
        alpha = self._params[2:2+self.p]
        beta  = self._params[2+self.p:]
        persistence = float(np.sum(alpha) + np.sum(beta))
        n, k = self._n, len(self._params)
        pnames = ['mu', 'omega'] + [f'alpha{i+1}' for i in range(self.p)] + \
                 [f'beta{j+1}' for j in range(self.q)]
        # 简化 SE：用对数似然数值二阶导
        try:
            se = self._hessian_se(self._residuals + self._params[0])
        except Exception:
            se = np.full(k, np.nan)
        z = self._params / np.where(se > 0, se, np.nan)
        p = 2 * stats.norm.sf(np.abs(z))
        coef_tbl = {
            nm: dict(coef=float(self._params[i]), se=float(se[i]),
                     z=float(z[i]), p_value=float(p[i]))
            for i, nm in enumerate(pnames)
        }
        return {
            "model": f"GARCH({self.p},{self.q})",
            "n_obs": n,
            "log_likelihood": self._loglik,
            "AIC": -2*self._loglik + 2*k,
            "BIC": -2*self._loglik + k*np.log(n),
            "persistence": persistence,
            "unconditional_variance": float(omega / max(1 - persistence, 1e-8)),
            "coefficients": coef_tbl,
        }

    def summary_table(self) -> pd.DataFrame:
        s = self.summary()
        rows = [{"parameter": nm, **vals} for nm, vals in s["coefficients"].items()]
        return pd.DataFrame(rows).set_index("parameter")


def stock_return_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """生成具有波动聚集性的合成日收益率（模拟沪深 300 风格）。

    真实参数：omega=0.00001, alpha1=0.10, beta1=0.85
    """
    rng = np.random.default_rng(seed)
    omega, alpha1, beta1, mu = 1e-5, 0.10, 0.85, 0.0003
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = omega / (1 - alpha1 - beta1)
    for t in range(1, n):
        sigma2[t] = omega + alpha1 * eps[t-1]**2 + beta1 * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    ret = mu + eps
    dates = pd.date_range('2018-01-01', periods=n, freq='B')
    return pd.DataFrame(dict(date=dates, ret=ret, sigma2_true=sigma2))


if __name__ == "__main__":
    import pprint
    df = stock_return_data()
    m = GARCH(p=1, q=1).fit(df['ret'])
    pprint.pprint({k: v for k, v in m.summary().items() if k != 'coefficients'})
    print(); print(m.summary_table().round(6))
    fwd = m.predict_variance(h=5)
    print(f"\n5 步前向波动率预测（年化）: {np.sqrt(fwd * 252).round(4)}")
