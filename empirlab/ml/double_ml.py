"""Double Machine Learning (DML) 最小模板。

此处为简化模板：定义类与接口，包含 docstring 和示例。
具体算法可在后续迭代中补全。
"""
import numpy as np


class DoubleML:
    """DoubleML 模型模板。

    API 与项目约定一致：`fit()`、`ate()`、`summary()`。
    """

    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, T):
        """拟合 DML：X 特征矩阵，y 结果，T 处理变量（0/1）。

        为模板，实际实现需使用交叉拟合等步骤。
        """
        self.is_fitted = True
        return self

    def ate(self):
        if not self.is_fitted:
            raise RuntimeError("请先 fit()")
        return 0.0

    def summary(self):
        return {"ate": self.ate()}


if __name__ == "__main__":
    import numpy as np

    np.random.seed(2)
    n = 200
    X = np.random.randn(n, 5)
    T = np.random.binomial(1, 0.5, size=n)
    y = 0.5 * T + X[:, 0] * 0.3 + np.random.randn(n) * 0.2

    model = DoubleML()
    model.fit(X, y, T)
    print("ATE：", model.ate())
"""
双重机器学习 (DoubleML)
=========================
来源参考：Chernozhukov et al. (2018) Double/Debiased ML
适用场景：高维控制变量下处理效应估计
Python 版本：3.10+
依赖：numpy >= 1.21 / pandas >= 1.3 / scikit-learn >= 1.3
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from empirlab.utils.metrics import calculate_metrics


class DoubleML:
    """简化版双重机器学习模型。"""

    def __init__(self):
        self.is_fitted = False
        self.theta_: float | None = None
        self._y_residual: np.ndarray | None = None
        self._d_residual: np.ndarray | None = None
        self._y_hat: np.ndarray | None = None

    def fit(self, X, y, d, **kwargs):
        """拟合 DoubleML。"""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        d_arr = np.asarray(d, dtype=float).reshape(-1)

        model_y = LinearRegression().fit(X_arr, y_arr)
        model_d = LinearRegression().fit(X_arr, d_arr)

        y_tilde = y_arr - model_y.predict(X_arr)
        d_tilde = d_arr - model_d.predict(X_arr)

        denom = float(np.dot(d_tilde, d_tilde))
        if denom <= 1e-12:
            raise RuntimeError("处理变量残差方差过小，无法稳定估计。")

        self.theta_ = float(np.dot(d_tilde, y_tilde) / denom)
        self._y_residual = y_tilde
        self._d_residual = d_tilde
        self._y_hat = self.theta_ * d_tilde
        self.is_fitted = True
        return self

    def predict(self, d):
        """基于处理变量残差做线性预测。"""
        if not self.is_fitted or self.theta_ is None:
            raise RuntimeError("请先调用 fit() 方法。")
        d_arr = np.asarray(d, dtype=float).reshape(-1)
        return self.theta_ * d_arr

    def summary(self):
        """返回处理效应摘要。"""
        if not self.is_fitted or self._y_residual is None or self._y_hat is None:
            raise RuntimeError("请先调用 fit() 方法。")
        metrics = calculate_metrics(self._y_residual, self._y_hat)
        return {
            "model": "DoubleML",
            "theta": self.theta_,
            "metrics": metrics,
            "n_observations": int(self._y_residual.shape[0]),
        }


if __name__ == "__main__":
    np.random.seed(42)
    n = 400
    X = np.random.randn(n, 5)
    d = 0.4 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n) * 0.3
    y = 1.2 * d + 0.5 * X[:, 0] - 0.2 * X[:, 2] + np.random.randn(n) * 0.5

    model = DoubleML()
    model.fit(X, y, d)
    result = model.summary()
    print("处理效应 theta:", result["theta"])
    print("拟合指标:", pd.Series(result["metrics"]).to_dict())
