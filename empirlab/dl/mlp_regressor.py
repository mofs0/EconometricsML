"""
多层感知机回归 (MLP Regressor)
===============================
来源参考：Goodfellow et al. (2016) Deep Learning
适用场景：非线性回归与高维特征拟合
Python 版本：3.10+
依赖：numpy >= 1.21 / pandas >= 1.3 / torch >= 2.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from empirlab.utils.metrics import calculate_metrics


class MlpRegressor:
    """基于 PyTorch 的最小 MLP 回归模型。"""

    def __init__(self, hidden_dim: int = 16, lr: float = 1e-2, epochs: int = 80, batch_size: int = 32):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_fitted = False
        self.model: nn.Module | None = None
        self._last_y_true: np.ndarray | None = None
        self._last_y_pred: np.ndarray | None = None

    def fit(self, X, y, **kwargs):
        """拟合 MLP 回归模型。"""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.model = nn.Sequential(
            nn.Linear(X_arr.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._last_y_true = y_arr.reshape(-1)
        self._last_y_pred = self.predict(X_arr)
        self.is_fitted = True
        return self

    def predict(self, X):
        """预测。"""
        if not self.is_fitted and self.model is None:
            raise RuntimeError("请先调用 fit() 方法。")
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            y_hat = self.model(torch.from_numpy(X_arr)).numpy().reshape(-1)
        return y_hat

    def summary(self):
        """返回模型统计摘要（字典格式）。"""
        if not self.is_fitted or self._last_y_true is None or self._last_y_pred is None:
            raise RuntimeError("请先调用 fit() 方法。")
        metrics = calculate_metrics(self._last_y_true, self._last_y_pred)
        return {
            "model": "MlpRegressor",
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "metrics": metrics,
            "n_observations": int(self._last_y_true.shape[0]),
        }


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    n = 300
    X = np.random.randn(n, 4)
    y = 1.2 * X[:, 0] ** 2 - 0.7 * X[:, 1] + 0.3 * X[:, 2] + np.random.randn(n) * 0.2

    model = MlpRegressor()
    model.fit(X, y)
    result = model.summary()
    print("模型摘要:", pd.Series(result["metrics"]).to_dict())
