"""断点回归设计 (Regression Discontinuity Design)
====================================================
来源参考：Lee & Lemieux (2010). "Regression Discontinuity Designs in Economics". JEL;
          Carpenter & Dobkin (2009). "The Effect of Alcohol Consumption on Mortality:
          Regression Discontinuity Evidence from the Minimum Drinking Age".
          AEJ: Applied Economics, 1(1): 164-182.
适用场景：处理状态由连续"运行变量"在阈值处的跳跃决定的准自然实验。
Python 版本：3.10+
依赖：numpy >= 1.24, pandas >= 1.5, scipy >= 1.10
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Literal, Optional
import numpy as np
import pandas as pd
from scipy import stats

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
from empirlab.utils.metrics import calculate_metrics


class RD:
    """Sharp 断点回归 (Sharp RDD).

    估计断点处的局部平均处理效应（LATE）：
        E[Y(1) - Y(0) | X = c]

    使用局部线性回归（矩形或三角核，带宽 h）：
        y = α + τ·D + β₁·x_c + β₂·D·x_c + ε
    其中 x_c = X - cutoff，D = 1{X >= cutoff}。

    参数
    ----
    cutoff    : float  断点值
    bandwidth : float or None  带宽（None 则用 Imbens-Kalyanaraman 启发式规则）
    kernel    : {'triangular', 'uniform'}  核函数
    poly      : int, default 1  多项式阶数（1=局部线性，2=局部二次）
    robust    : bool, default True

    示例
    ----
    >>> from empirlab.traditional.rd import RD, drinking_data
    >>> df = drinking_data()
    >>> m = RD(cutoff=21.0, bandwidth=2.0)
    >>> m.fit(df['age'], df['mortality'])
    >>> print(m.summary_table().round(4))
    """

    def __init__(self, cutoff: float = 0.0,
                 bandwidth: Optional[float] = None,
                 kernel: Literal['triangular','uniform'] = 'triangular',
                 poly: int = 1,
                 robust: bool = True):
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.poly = poly
        self.robust = robust
        self.is_fitted = False
        self.coef_: Optional[np.ndarray] = None
        self._tau: Optional[float] = None
        self._se_tau: Optional[float] = None
        self._bw_used: Optional[float] = None

    def _ik_bandwidth(self, x_c, y, D):
        """Imbens-Kalyanaraman (2012) 启发式带宽选择（简化版）。"""
        h_pilot = 1.84 * np.std(x_c) * len(x_c) ** (-0.2)
        mask = np.abs(x_c) <= h_pilot
        if mask.sum() < 10:
            return float(np.std(x_c))
        m2_r = np.polyfit(x_c[mask & (D == 1)], y[mask & (D == 1)], 2)
        m2_l = np.polyfit(x_c[mask & (D == 0)], y[mask & (D == 0)], 2)
        n = len(x_c)
        h_opt = 3.56 * (np.std(y[mask]) ** 2 /
                        (n * (m2_r[0] - m2_l[0]) ** 2 + 1e-10)) ** (1/5)
        return max(float(h_opt), 0.1 * np.std(x_c))

    def _kernel_weights(self, x_c, h):
        u = x_c / h
        if self.kernel == 'triangular':
            w = np.maximum(1 - np.abs(u), 0)
        else:
            w = (np.abs(u) <= 1).astype(float)
        return w

    def _build_X(self, x_c, D):
        cols = [np.ones_like(x_c), D]
        for p in range(1, self.poly + 1):
            cols.append(x_c ** p)
            cols.append(D * x_c ** p)
        return np.column_stack(cols)

    def fit(self, running_var, outcome) -> "RD":
        """拟合 Sharp RDD。

        参数
        ----
        running_var : array-like  运行变量（连续）
        outcome     : array-like  因变量
        """
        x = np.asarray(running_var, dtype=float)
        y = np.asarray(outcome, dtype=float)
        x_c = x - self.cutoff
        D = (x >= self.cutoff).astype(float)

        h = self.bandwidth if self.bandwidth is not None else self._ik_bandwidth(x_c, y, D)
        self._bw_used = h

        w = self._kernel_weights(x_c, h)
        mask = w > 0
        x_c_m, y_m, D_m, w_m = x_c[mask], y[mask], D[mask], w[mask]

        X = self._build_X(x_c_m, D_m)
        W = np.diag(w_m)
        XtWX = X.T @ W @ X
        XtWy = X.T @ (w_m * y_m)
        beta = np.linalg.pinv(XtWX) @ XtWy
        resid = y_m - X @ beta

        # 标准误
        n, k = X.shape
        if self.robust:
            Xe = (X * w_m[:, None]) * resid[:, None]
            meat = Xe.T @ Xe
            cov = np.linalg.pinv(XtWX) @ meat @ np.linalg.pinv(XtWX)
        else:
            sigma2 = np.sum(w_m * resid**2) / (n - k)
            cov = sigma2 * np.linalg.pinv(XtWX)

        self.coef_ = beta
        self._tau = float(beta[1])           # D 系数 = LATE
        self._se_tau = float(np.sqrt(cov[1, 1]))
        self._n_used = int(mask.sum())
        self._n_left  = int(((x_c < 0) & mask).sum())
        self._n_right = int(((x_c >= 0) & mask).sum())
        self.is_fitted = True
        return self

    def summary(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        tau, se = self._tau, self._se_tau
        t = tau / se
        p = float(2 * stats.norm.sf(abs(t)))
        cv = stats.norm.ppf(0.975)
        return {
            "model": "Sharp-RDD",
            "cutoff": self.cutoff,
            "bandwidth": round(self._bw_used, 4),
            "kernel": self.kernel,
            "poly": self.poly,
            "n_obs_used": self._n_used,
            "n_left": self._n_left,
            "n_right": self._n_right,
            "LATE": {
                "coef": tau, "se": se, "t": float(t),
                "p_value": p,
                "ci_lower": float(tau - cv * se),
                "ci_upper": float(tau + cv * se),
            },
            "note": "LATE = 断点处局部平均处理效应 E[Y(1)-Y(0)|X=cutoff]",
        }

    def summary_table(self) -> pd.DataFrame:
        s = self.summary()
        return pd.DataFrame([{"parameter": "LATE (D)", **s["LATE"]}]).set_index("parameter")


def drinking_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Carpenter & Dobkin (2009) 风格：最低饮酒年龄对死亡率的影响。

    DGP（断点 c = 21 岁）：
        mortality = 28 + 7.5*D + 0.2*(age-21) + 0.4*D*(age-21) + eps
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(19, 23, n)
    D = (age >= 21.0).astype(float)
    x_c = age - 21.0
    eps = rng.normal(0, 3.0, n)
    mortality = 28.0 + 7.5 * D + 0.2 * x_c + 0.4 * D * x_c + eps
    return pd.DataFrame(dict(age=age, D=D, mortality=mortality))


if __name__ == "__main__":
    import pprint
    df = drinking_data()
    m = RD(cutoff=21.0, bandwidth=2.0, kernel='triangular')
    m.fit(df['age'], df['mortality'])
    pprint.pprint(m.summary())
    print(); print(m.summary_table().round(4))
    print(f"\n真实 LATE = 7.5，估计值 = {m.summary()['LATE']['coef']:.4f}")
