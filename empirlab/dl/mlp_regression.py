"""多层感知机回归（MLP）
========================
适用场景：
  - 非线性截面数据回归（替代 OLS 的深度学习基准）
  - 特征工程后的预测任务
  - 与 LASSO/RF 对比的基准深度模型

设计原则：
  - 支持任意深度和宽度
  - BatchNorm + Dropout 防过拟合
  - EarlyStopping（验证集 patience）
  - sklearn 兼容接口（fit / predict）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _MLP(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")
        super().__init__()
        layers: list = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPRegressor:
    """多层感知机回归（PyTorch 实现）。

    参数
    ----
    hidden_dims  : list[int], default [128, 64, 32]  —— 隐藏层维度
    dropout      : float, default 0.2
    lr           : float, default 1e-3
    epochs       : int, default 200
    batch_size   : int, default 128
    patience     : int, default 20  —— EarlyStopping
    val_frac     : float, default 0.1  —— 验证集比例
    device       : str, default "auto"

    示例
    ----
    >>> import numpy as np
    >>> from empirlab.dl import MLPRegressor
    >>> X = np.random.randn(500, 10)
    >>> y = X[:, 0]**2 - X[:, 1]*X[:, 2] + np.random.randn(500)*0.3
    >>> model = MLPRegressor(epochs=50).fit(X, y)
    >>> print(f"Train RMSE: {model.train_rmse_:.4f}")
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 128,
        patience: int = 20,
        val_frac: float = 0.1,
        device: str = "auto",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: Optional[_MLP] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.train_rmse_: Optional[float] = None
        self.best_epoch_: int = 0
        self.history_: dict = {"train": [], "val": []}

    def fit(self, X, y, verbose: bool = False) -> "MLPRegressor":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        n = X_arr.shape[0]

        # 标准化
        self._x_mean = X_arr.mean(0)
        self._x_std = X_arr.std(0) + 1e-8
        self._y_mean = float(y_arr.mean())
        self._y_std = float(y_arr.std()) + 1e-8
        X_norm = (X_arr - self._x_mean) / self._x_std
        y_norm = (y_arr - self._y_mean) / self._y_std

        # 训练/验证拆分
        n_val = max(1, int(n * self.val_frac))
        idx = np.random.permutation(n)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        X_tr = torch.tensor(X_norm[tr_idx]).to(self.device)
        y_tr = torch.tensor(y_norm[tr_idx]).to(self.device)
        X_val = torch.tensor(X_norm[val_idx]).to(self.device)
        y_val = torch.tensor(y_norm[val_idx]).to(self.device)

        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=self.batch_size, shuffle=True)

        self._model = _MLP(X_arr.shape[1], self.hidden_dims, 1, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val, best_state, wait = np.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            self._model.train()
            tl = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                tl += loss.item()
            tl /= len(loader)

            self._model.eval()
            with torch.no_grad():
                vl = float(criterion(self._model(X_val), y_val).item())

            self.history_["train"].append(tl)
            self.history_["val"].append(vl)

            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}  train={tl:.4f}  val={vl:.4f}")

            if vl < best_val:
                best_val, best_state, wait = vl, self._model.state_dict(), 0
                self.best_epoch_ = epoch
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print(f"  EarlyStopping at epoch {epoch}")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        # 训练集 RMSE（原始量纲）
        self._model.eval()
        with torch.no_grad():
            tr_pred = (self._model(X_tr).cpu().numpy() * self._y_std + self._y_mean)
        tr_true = y_arr[tr_idx]
        self.train_rmse_ = float(np.sqrt(np.mean((tr_pred - tr_true) ** 2)))
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("请先调用 fit()。")
        X_arr = (np.asarray(X, dtype=np.float32) - self._x_mean) / self._x_std
        x = torch.tensor(X_arr).to(self.device)
        self._model.eval()
        with torch.no_grad():
            pred_norm = self._model(x).cpu().numpy()
        return pred_norm * self._y_std + self._y_mean


if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        print("PyTorch 未安装，跳过示例。")
    else:
        np.random.seed(42)
        X = np.random.randn(600, 8)
        y = X[:, 0] ** 2 - 1.5 * X[:, 1] * X[:, 2] + np.sin(X[:, 3]) + np.random.randn(600) * 0.3

        model = MLPRegressor(hidden_dims=[64, 32], epochs=100, patience=15)
        model.fit(X, y, verbose=True)
        print(f"\nBest epoch: {model.best_epoch_}, Train RMSE: {model.train_rmse_:.4f}")
