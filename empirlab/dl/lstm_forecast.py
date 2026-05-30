"""LSTM 时间序列预测
====================
来源参考：Hochreiter, S., & Schmidhuber, J. (1997).
          Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

适用场景：
  - 金融时间序列预测（股价、利率、汇率）
  - 宏观经济变量预测（GDP、CPI）
  - 多步预测与单步预测

依赖：PyTorch >= 2.0

说明：
  若环境无 GPU，自动使用 CPU。
  建议序列长度 lookback ≥ 20，batch_size = 64。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

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


def make_sequences(
    series: np.ndarray,
    lookback: int = 20,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """将一维时间序列切成 (X, y) 监督学习格式。

    参数
    ----
    series   : shape (T,) 或 (T, n_features)
    lookback : 输入窗口长度
    horizon  : 预测步长（默认 1，单步预测）

    返回
    ----
    X : shape (N, lookback, n_features)
    y : shape (N, horizon)
    """
    arr = np.asarray(series, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    T, F = arr.shape
    xs, ys = [], []
    for i in range(T - lookback - horizon + 1):
        xs.append(arr[i: i + lookback])          # (lookback, F)
        ys.append(arr[i + lookback: i + lookback + horizon, 0])  # (horizon,)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class _LSTMModel(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMForecaster:
    """LSTM 时间序列预测器。

    参数
    ----
    lookback   : int, default 20  —— 回望窗口
    horizon    : int, default 1   —— 预测步长
    hidden     : int, default 64  —— LSTM 隐藏层维度
    num_layers : int, default 2   —— LSTM 层数
    dropout    : float, default 0.1
    lr         : float, default 1e-3
    epochs     : int, default 100
    batch_size : int, default 64
    device     : str, default "auto"  —— "cpu"/"cuda"/"auto"

    示例
    ----
    >>> import numpy as np
    >>> from empirlab.dl import LSTMForecaster, make_sequences
    >>> t = np.linspace(0, 20*np.pi, 2000)
    >>> series = np.sin(t) + np.random.normal(0, 0.1, 2000)
    >>> model = LSTMForecaster(lookback=30, epochs=30).fit(series)
    >>> pred = model.predict(series[-30:])
    """

    def __init__(
        self,
        lookback: int = 20,
        horizon: int = 1,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        device: str = "auto",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")
        self.lookback = lookback
        self.horizon = horizon
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: Optional[_LSTMModel] = None
        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0
        self.train_losses_: list = []

    def fit(self, series, verbose: bool = False) -> "LSTMForecaster":
        """训练 LSTM 模型。

        参数
        ----
        series  : array-like shape (T,) 或 (T, n_features)
        verbose : 每 10 epoch 打印一次 loss
        """
        arr = np.asarray(series, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_features = arr.shape[1]

        # Z-score 归一化
        self._scaler_mean = float(arr[:, 0].mean())
        self._scaler_std = float(arr[:, 0].std()) + 1e-8
        arr_norm = arr.copy()
        arr_norm[:, 0] = (arr[:, 0] - self._scaler_mean) / self._scaler_std

        X, y = make_sequences(arr_norm, self.lookback, self.horizon)
        X_t = torch.tensor(X).to(self.device)
        y_t = torch.tensor(y).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = _LSTMModel(
            n_features, self.hidden, self.num_layers, self.horizon, self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            self.train_losses_.append(avg)
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        return self

    def predict(self, context) -> np.ndarray:
        """单步 / 多步预测。

        参数
        ----
        context : 最近 lookback 个时间步的原始值，shape (lookback,) 或 (lookback, F)

        返回
        ----
        np.ndarray, shape (horizon,)，已还原到原始量纲
        """
        if self._model is None:
            raise RuntimeError("请先调用 fit()。")
        arr = np.asarray(context, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr_norm = arr.copy()
        arr_norm[:, 0] = (arr[:, 0] - self._scaler_mean) / self._scaler_std

        x = torch.tensor(arr_norm[None]).to(self.device)  # (1, lookback, F)
        self._model.eval()
        with torch.no_grad():
            pred_norm = self._model(x).cpu().numpy().ravel()

        return pred_norm * self._scaler_std + self._scaler_mean


if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        print("PyTorch 未安装，跳过示例。")
    else:
        np.random.seed(0)
        t = np.linspace(0, 40 * np.pi, 4000)
        series = np.sin(t) + 0.1 * np.random.randn(4000)

        model = LSTMForecaster(lookback=30, horizon=1, epochs=30, hidden=32)
        model.fit(series, verbose=True)

        context = series[-30:]
        pred = model.predict(context)
        true_next = series[-1 + 1] if len(series) > 4000 else "无真值（末端）"
        print(f"\n预测下一步：{pred[0]:.4f}（真值约 {np.sin(t[-1] + t[1] - t[0]):.4f}）")
