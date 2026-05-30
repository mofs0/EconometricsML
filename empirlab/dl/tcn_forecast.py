"""时序卷积网络（TCN）预测
============================
来源参考：Bai, S., Kolter, J. Z., & Koltun, V. (2018).
          An empirical evaluation of generic convolutional and recurrent
          networks for sequence modeling. arXiv:1803.01271.

适用场景：
  - 长序列依赖建模（感受野随层数指数增长）
  - 比 LSTM 训练更稳定、支持并行计算
  - 金融高频数据预测

核心思想：
  膨胀因果卷积（Dilated Causal Convolution）
    感受野 = 2^(num_levels) * kernel_size
  残差连接（ResBlock）保证梯度流动
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from .lstm_forecast import make_sequences

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _ResBlock(nn.Module if _TORCH_AVAILABLE else object):
    """TCN 残差块：膨胀因果卷积 + LayerNorm + 残差连接。"""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self._pad = pad

    def _trim(self, x, pad):
        return x[:, :, :-pad].contiguous() if pad > 0 else x

    def forward(self, x):
        # x: (B, C, L)
        out = self._trim(self.conv1(x), self._pad)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.drop(out)

        out = self._trim(self.conv2(out), self._pad)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.drop(out)

        res = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + res)


class _TCNModel(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, in_ch, n_channels, kernel_size, dropout, output_size):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")
        super().__init__()
        layers = []
        ch = in_ch
        for i, c in enumerate(n_channels):
            layers.append(_ResBlock(ch, c, kernel_size, dilation=2 ** i, dropout=dropout))
            ch = c
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(n_channels[-1], output_size)

    def forward(self, x):
        # x: (B, L, F) → (B, F, L)
        out = self.net(x.permute(0, 2, 1))
        return self.fc(out[:, :, -1])


class TCNForecaster:
    """TCN 时间序列预测器。

    参数
    ----
    lookback    : int, default 30
    horizon     : int, default 1
    n_channels  : list[int], default [32, 32, 32]  —— 各 ResBlock 通道数
    kernel_size : int, default 3
    dropout     : float, default 0.1
    lr          : float, default 1e-3
    epochs      : int, default 80
    batch_size  : int, default 64

    示例
    ----
    >>> import numpy as np
    >>> from empirlab.dl import TCNForecaster
    >>> series = np.sin(np.linspace(0, 20*np.pi, 2000))
    >>> model = TCNForecaster(lookback=30, epochs=20).fit(series)
    """

    def __init__(
        self,
        lookback: int = 30,
        horizon: int = 1,
        n_channels=None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 64,
        device: str = "auto",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")
        self.lookback = lookback
        self.horizon = horizon
        self.n_channels = n_channels or [32, 32, 32]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: Optional[_TCNModel] = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self.train_losses_: list = []

    def fit(self, series, verbose: bool = False) -> "TCNForecaster":
        arr = np.asarray(series, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_features = arr.shape[1]

        self._mean = float(arr[:, 0].mean())
        self._std = float(arr[:, 0].std()) + 1e-8
        arr_norm = arr.copy()
        arr_norm[:, 0] = (arr[:, 0] - self._mean) / self._std

        X, y = make_sequences(arr_norm, self.lookback, self.horizon)
        X_t = torch.tensor(X).to(self.device)
        y_t = torch.tensor(y).to(self.device)

        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=self.batch_size, shuffle=True)

        self._model = _TCNModel(
            n_features, self.n_channels, self.kernel_size, self.dropout, self.horizon
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item()
            avg = total / len(loader)
            self.train_losses_.append(avg)
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")
        return self

    def predict(self, context) -> np.ndarray:
        """预测，context 为最近 lookback 步的原始值。"""
        if self._model is None:
            raise RuntimeError("请先调用 fit()。")
        arr = np.asarray(context, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr_norm = arr.copy()
        arr_norm[:, 0] = (arr[:, 0] - self._mean) / self._std

        x = torch.tensor(arr_norm[None]).to(self.device)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(x).cpu().numpy().ravel()
        return pred * self._std + self._mean


if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        print("PyTorch 未安装，跳过示例。")
    else:
        np.random.seed(1)
        t = np.linspace(0, 40 * np.pi, 4000)
        series = np.sin(t) + 0.1 * np.random.randn(4000)

        model = TCNForecaster(lookback=30, epochs=20, n_channels=[16, 32])
        model.fit(series, verbose=True)
        pred = model.predict(series[-30:])
        print(f"\n预测下一步：{pred[0]:.4f}（真值约 {np.sin(t[-1] + t[1] - t[0]):.4f}）")
