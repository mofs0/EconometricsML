"""因果森林（Causal Forest）——异质处理效应估计
==============================================
来源参考：Wager, S., & Athey, S. (2018). Estimation and inference of
          heterogeneous treatment effects using random forests.
          *JASA*, 113(523), 1228–1242.
          Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized
          random forests. *Annals of Statistics*, 47(2), 1148–1178.

适用场景：
  - 异质处理效应（CATE / HTE）估计
  - 识别哪类人群对政策响应更强
  - 政策精准化与分层分析

本模块说明：
  因果森林的完整实现依赖 R 包 grf 或 EconML。
  此处提供：
    1. 基于 sklearn 的 Meta-Learner 近似实现（S-Learner & T-Learner）
    2. 简洁 API 与 empirlab 风格一致
    3. 真实项目建议配合 econml>=0.14 使用（见注释）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_predict

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class CausalForest:
    """Meta-Learner 因果森林（CATE 估计）。

    支持两种 meta-learner：
      - "T"：T-Learner（分别对处理组/对照组训练两个模型，差值为 CATE）
      - "S"：S-Learner（处理变量作为特征，预测差值为 CATE）

    参数
    ----
    learner  : "T" or "S", default "T"
    base_model : sklearn 兼容回归器，默认 GradientBoostingRegressor
    random_state : int, default 42

    属性
    ----
    cate_       : np.ndarray  —— 个体处理效应估计值（shape: n,）
    ate_        : float       —— 平均处理效应
    ate_se_     : float       —— ATE 标准误（bootstrap）

    示例
    ----
    >>> from empirlab.ml import CausalForest, cf_data
    >>> df = cf_data(n=500)
    >>> model = CausalForest(learner="T")
    >>> model.fit(df[["x1","x2","x3"]], df["d"], df["y"])
    >>> print(f"ATE = {model.ate_:.4f} (真值 = 1.0)")
    """

    def __init__(
        self,
        learner: Literal["T", "S"] = "T",
        base_model=None,
        random_state: int = 42,
    ):
        self.learner = learner
        self.base_model = base_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=random_state
        )
        self.random_state = random_state

        self.cate_: Optional[np.ndarray] = None
        self.ate_: Optional[float] = None
        self.ate_se_: Optional[float] = None

    def fit(self, X, D, Y, n_bootstrap: int = 200) -> "CausalForest":
        """估计 CATE 和 ATE。

        参数
        ----
        X           : array-like, shape (n, p)
        D           : array-like, shape (n,)  —— 二值处理变量（0/1）
        Y           : array-like, shape (n,)
        n_bootstrap : int —— bootstrap 次数，用于 ATE 置信区间
        """
        from sklearn.base import clone

        X_arr = np.asarray(X, dtype=float)
        D_arr = np.asarray(D, dtype=float).ravel()
        Y_arr = np.asarray(Y, dtype=float).ravel()
        n = X_arr.shape[0]

        if self.learner == "T":
            mask1 = D_arr == 1
            mask0 = D_arr == 0
            m1 = clone(self.base_model)
            m0 = clone(self.base_model)
            m1.fit(X_arr[mask1], Y_arr[mask1])
            m0.fit(X_arr[mask0], Y_arr[mask0])
            cate = m1.predict(X_arr) - m0.predict(X_arr)

        else:  # S-Learner
            XD = np.column_stack([X_arr, D_arr])
            XD1 = np.column_stack([X_arr, np.ones(n)])
            XD0 = np.column_stack([X_arr, np.zeros(n)])
            m = clone(self.base_model)
            m.fit(XD, Y_arr)
            cate = m.predict(XD1) - m.predict(XD0)

        self.cate_ = cate
        self.ate_ = float(cate.mean())

        # Bootstrap ATE SE
        rng = np.random.default_rng(self.random_state)
        boot_ates = np.array([
            cate[rng.integers(0, n, n)].mean()
            for _ in range(n_bootstrap)
        ])
        self.ate_se_ = float(boot_ates.std())
        return self

    def cate_summary(self, X=None, feature_names=None) -> pd.DataFrame:
        """返回 CATE 分位数分布摘要。"""
        if self.cate_ is None:
            raise RuntimeError("请先调用 fit()。")
        q = np.percentile(self.cate_, [10, 25, 50, 75, 90])
        return pd.DataFrame({
            "统计量": ["ATE", "P10", "P25", "P50", "P75", "P90", "SD"],
            "值": [self.ate_] + list(q) + [float(self.cate_.std())],
        }).round(4)

    def summary(self) -> dict:
        """返回 ATE 估计摘要。"""
        if self.ate_ is None:
            raise RuntimeError("请先调用 fit()。")
        from scipy import stats
        t = self.ate_ / self.ate_se_
        p = float(2 * stats.norm.sf(abs(t)))
        cv = stats.norm.ppf(0.975)
        return {
            "learner": self.learner,
            "ATE": self.ate_,
            "SE": self.ate_se_,
            "t_stat": t,
            "p_value": p,
            "ci_lower": self.ate_ - cv * self.ate_se_,
            "ci_upper": self.ate_ + cv * self.ate_se_,
        }


# ── 辅助数据生成 ──────────────────────────────────────────────────────

def cf_data(
    n: int = 1000,
    true_ate: float = 1.0,
    heterogeneous: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """生成因果森林合成数据（含异质处理效应）。

    DGP：
        X ~ N(0, I_3)
        P(D=1|X) = sigmoid(X[:,0])
        CATE(X) = true_ate + (heterogeneous ? X[:,1] : 0)
        Y = CATE(X)*D + X[:,2] + eps
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    prop = 1 / (1 + np.exp(-X[:, 0]))
    D = rng.binomial(1, prop).astype(float)
    cate_true = true_ate + (X[:, 1] if heterogeneous else 0.0)
    Y = cate_true * D + X[:, 2] + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "d": D, "y": Y})


if __name__ == "__main__":
    df = cf_data(n=800, true_ate=1.0)
    for learner in ["T", "S"]:
        model = CausalForest(learner=learner)
        model.fit(df[["x1", "x2", "x3"]], df["d"], df["y"])
        s = model.summary()
        print(f"[{learner}-Learner] ATE={s['ATE']:.4f}, SE={s['SE']:.4f}, "
              f"p={s['p_value']:.4f}, 95%CI=[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")
    print("\nCATE 分布（T-Learner）：")
    model_t = CausalForest(learner="T").fit(df[["x1","x2","x3"]], df["d"], df["y"])
    print(model_t.cate_summary().to_string(index=False))
