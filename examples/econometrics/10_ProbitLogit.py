"""
Probit / Logit 示例（中英）

说明：展示 Logit 和 Probit 的简单拟合示例。
Usage: python examples/econometrics/10_ProbitLogit.py

依赖：statsmodels, numpy
"""

import numpy as np
import statsmodels.api as sm

def run_example():
    np.random.seed(8)
    X = np.random.randn(200,2)
    beta = np.array([1.0, -1.0])
    lin = X.dot(beta)
    p = 1 / (1 + np.exp(-lin))
    y = (np.random.rand(200) < p).astype(int)
    res = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    print(res.summary())

if __name__ == '__main__':
    run_example()
