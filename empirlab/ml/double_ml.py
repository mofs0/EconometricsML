"""Double / Debiased Machine Learning (DML)
==========================================
来源参考：Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
          Hansen, C., Newey, W., & Robins, J. (2018).
          Double/debiased machine learning for treatment and structural parameters.
          *The Econometrics Journal*, 21(1), C1–C68.

适用场景：处理变量（D）和结果变量（Y）均受高维混淆变量（X）影响时，
         利用机器学习部分线性化以估计处理效应 θ，同时避免正则化偏误。

模型设定：
    Y = θ·D + g(X) + ε，E[ε|D,X] = 0
    D = m(X) + v，        E[v|X] = 0

估计流程（Cross-fitting，K-fold）：
    1. 用 ML 模型拟合 Ŷ = g_hat(X)，残差 Ỹ = Y - Ŷ
    2. 用 ML 模型拟合 D̂ = m_hat(X)，残差 D̃ = D - D̂
    3. OLS 估计：θ = (D̃'D̃)^{-1} D̃'Ỹ，Neyman 正交得分保证 √n 一致性
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class DoubleML:
    """Double/Debiased Machine Learning 处理效应估计器。

    参数
    ----
    ml_y : sklearn 兼容回归器，默认 GradientBoostingRegressor
        用于拟合 E[Y|X]（结果方程的 nuisance 函数）。
    ml_d : sklearn 兼容回归器，默认 RidgeCV
        用于拟合 E[D|X]（处理方程的 nuisance 函数）。
    n_folds : int, default 5
        Cross-fitting 的折数。
    random_state : int, default 42

    属性
    ----
    theta_    : float —— 处理效应点估计
    se_       : float —— 标准误
    t_stat_   : float —— t 统计量
    p_value_  : float —— p 值（双尾）
    ci_       : tuple(float, float) —— 95% 置信区间

    示例
    ----
    >>> from empirlab.ml import DoubleML, dml_data
    >>> df = dml_data(n=500)
    >>> model = DoubleML().fit(df[["x1","x2","x3","x4","x5"]], df["d"], df["y"])
    >>> print(f"θ = {model.theta_:.4f}, SE = {model.se_:.4f}, p = {model.p_value_:.4f}")
    """

    def __init__(
        self,
        ml_y=None,
        ml_d=None,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.ml_y = ml_y or GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=random_state
        )
        self.ml_d = ml_d or RidgeCV(alphas=[0.1, 1.0, 10.0])
        self.n_folds = n_folds
        self.random_state = random_state

        self.theta_: Optional[float] = None
        self.se_: Optional[float] = None
        self.t_stat_: Optional[float] = None
        self.p_value_: Optional[float] = None
        self.ci_: Optional[tuple] = None
        self._resid_y: Optional[np.ndarray] = None
        self._resid_d: Optional[np.ndarray] = None

    def fit(self, X, D, Y) -> "DoubleML":
        """估计处理效应 θ。

        参数
        ----
        X : array-like, shape (n, p)  —— 混淆变量矩阵
        D : array-like, shape (n,)    —— 处理变量（连续或二值）
        Y : array-like, shape (n,)    —— 结果变量

        返回
        ----
        self
        """
        X_arr = np.asarray(X, dtype=float)
        D_arr = np.asarray(D, dtype=float).ravel()
        Y_arr = np.asarray(Y, dtype=float).ravel()
        n = X_arr.shape[0]

        resid_y = np.zeros(n)
        resid_d = np.zeros(n)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            Y_tr, Y_val = Y_arr[train_idx], Y_arr[val_idx]
            D_tr, D_val = D_arr[train_idx], D_arr[val_idx]

            # nuisance 1：Y ~ X
            ml_y_fold = clone(self.ml_y)
            ml_y_fold.fit(X_tr, Y_tr)
            resid_y[val_idx] = Y_val - ml_y_fold.predict(X_val)

            # nuisance 2：D ~ X
            ml_d_fold = clone(self.ml_d)
            ml_d_fold.fit(X_tr, D_tr)
            resid_d[val_idx] = D_val - ml_d_fold.predict(X_val)

        # Neyman 正交估计 θ
        denom = float(np.sum(resid_d ** 2))
        theta = float(np.sum(resid_d * resid_y) / denom)

        # 方差估计（score 方差）
        psi = resid_d * (resid_y - theta * resid_d)
        var = float(np.sum(psi ** 2) / denom ** 2)
        se = float(np.sqrt(var / n))

        from scipy import stats
        t = theta / se
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
        cv = stats.norm.ppf(0.975)

        self.theta_ = theta
        self.se_ = se
        self.t_stat_ = t
        self.p_value_ = p
        self.ci_ = (theta - cv * se, theta + cv * se)
        self._resid_y = resid_y
        self._resid_d = resid_d
        return self

    def summary(self) -> pd.DataFrame:
        """返回处理效应估计结果表。"""
        if self.theta_ is None:
            raise RuntimeError("请先调用 fit()。")
        return pd.DataFrame(
            [{
                "theta": self.theta_,
                "se": self.se_,
                "t_stat": self.t_stat_,
                "p_value": self.p_value_,
                "ci_lower": self.ci_[0],
                "ci_upper": self.ci_[1],
            }]
        ).round(6)


# ── 辅助数据生成 ──────────────────────────────────────────────────────

def dml_data(
    n: int = 1000,
    p: int = 5,
    true_theta: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """生成 DML 合成数据集。

    DGP：
        X_i ~ N(0, I_p)
        D_i = X_i @ alpha + v_i,  v ~ N(0,1)
        Y_i = theta*D_i + X_i @ beta + eps_i, eps ~ N(0,1)

    参数
    ----
    n          : 样本量
    p          : 混淆变量维度
    true_theta : 真实处理效应
    seed       : 随机种子

    返回
    ----
    pd.DataFrame，含 x1…xp, d, y 列
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    alpha = rng.standard_normal(p) * 0.5
    beta = rng.standard_normal(p) * 0.3
    D = X @ alpha + rng.standard_normal(n)
    Y = true_theta * D + X @ beta + rng.standard_normal(n)
    cols = {f"x{i+1}": X[:, i] for i in range(p)}
    cols.update({"d": D, "y": Y})
    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = dml_data(n=800, true_theta=0.5)
    model = DoubleML(n_folds=5).fit(
        df[[c for c in df.columns if c.startswith("x")]],
        df["d"], df["y"]
    )
    print("DML 处理效应估计（真值 θ = 0.5）")
    print(model.summary().to_string(index=False))
