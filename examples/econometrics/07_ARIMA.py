"""
ARIMA 示例（中英）

说明：演示如何使用 statsmodels 的 ARIMA 建模与预测。
Usage: python examples/econometrics/07_ARIMA.py

依赖：statsmodels, numpy
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def run_example():
    np.random.seed(5)
    y = np.cumsum(np.random.randn(200))  # 随机游走样本
    model = ARIMA(y, order=(1,1,1)).fit()
    print(model.summary())

if __name__ == '__main__':
    run_example()
