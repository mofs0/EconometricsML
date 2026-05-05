"""
VAR 示例（中英）

说明：演示 statsmodels VAR 的拟合与脉冲响应分析。
Usage: python examples/econometrics/08_VAR.py

依赖：statsmodels, numpy, pandas
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

def run_example():
    np.random.seed(6)
    t = 200
    x1 = np.cumsum(np.random.randn(t))
    x2 = np.cumsum(np.random.randn(t))
    data = pd.DataFrame({'x1':x1,'x2':x2})
    model = VAR(data)
    res = model.fit(1)
    print(res.summary())

if __name__ == '__main__':
    run_example()
