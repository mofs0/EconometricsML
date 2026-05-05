"""
Difference-in-Differences (DiD) 示例（中英）

说明：展示 DiD 的回归实现与使用方法示例。
Usage: python examples/econometrics/05_DiD.py

依赖：statsmodels, pandas, numpy
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def run_example():
    np.random.seed(4)
    n = 200
    treated = np.concatenate([np.zeros(n//2), np.ones(n//2)])
    post = np.concatenate([np.zeros(n//4), np.ones(3*n//4)])
    # 构造处理效应
    y = 1.0 + 2.0 * treated * post + 0.5 * treated + 0.3 * post + np.random.randn(n)
    df = pd.DataFrame({'y':y, 'treated':treated, 'post':post, 'id':np.arange(n)})
    res = smf.ols('y ~ treated * post + C(id)', data=df).fit()
    print(res.summary())

if __name__ == '__main__':
    run_example()
