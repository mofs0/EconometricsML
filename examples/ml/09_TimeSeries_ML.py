"""
时间序列机器学习示例（滑动窗口特征）

Usage: python examples/ml/09_TimeSeries_ML.py
依赖：pandas, sklearn, numpy
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def run_example():
    np.random.seed(8)
    y = np.cumsum(np.random.randn(300))
    df = pd.DataFrame({'y':y})
    df['lag1'] = df['y'].shift(1)
    df['lag2'] = df['y'].shift(2)
    df = df.dropna()
    X = df[['lag1','lag2']].values
    model = LinearRegression().fit(X, df['y'].values)
    print('coeffs:', model.coef_)

if __name__ == '__main__':
    run_example()
