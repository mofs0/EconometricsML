"""
IV / 2SLS 示例（中英双语注释）

说明：使用工具变量的两阶段最小二乘法示例。
Usage: python examples/econometrics/02_IV.py

数学要点：第一阶段 X = Z pi + v；第二阶段使用 P_Z = Z (Z'Z)^{-1} Z'

依赖：linearmodels, numpy, pandas
"""

import numpy as np
from linearmodels.iv import IV2SLS

def run_example(n=200):
    np.random.seed(1)
    z = np.random.randn(n,1)
    v = np.random.randn(n,1)
    x = 0.8 * z + v
    y = 2.0 * x.squeeze() + np.random.randn(n)
    # 简单包装为 pandas DataFrame 由 linearmodels 接受
    import pandas as pd
    df = pd.DataFrame({'y':y, 'x':x.squeeze(), 'z':z.squeeze()})
    res = IV2SLS(dependent=df['y'], exog=pd.DataFrame({'const':1}), endog=df['x'], instruments=df['z']).fit()
    print(res)

if __name__ == '__main__':
    run_example()
