"""empirlab.dl —— 深度学习时序与回归子包。

模块列表
--------
lstm_forecast  : LSTM 时间序列预测（Hochreiter & Schmidhuber 1997）
tcn_forecast   : 时序卷积网络 TCN（Bai et al. 2018）
mlp_regression : 多层感知机回归（基准深度学习模型）
"""

from .lstm_forecast import LSTMForecaster, make_sequences
from .tcn_forecast import TCNForecaster
from .mlp_regression import MLPRegressor

__all__ = [
    "LSTMForecaster", "make_sequences",
    "TCNForecaster",
    "MLPRegressor",
]
