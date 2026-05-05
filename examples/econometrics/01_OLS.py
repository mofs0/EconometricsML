"""
OLS 示例（中英双语注释）
说明（中文）：
- 文件包含 OLS 理论简介、数学公式和一个最小可运行示例。
- 如何使用：在命令行运行 `python examples/econometrics/01_OLS.py`。

Description (English):
- OLS example with theory, formula and a runnable example.
- Usage: `python examples/econometrics/01_OLS.py`.

理论/数学公式：
# 线性模型 y = X beta + eps
# OLS 解: beta_hat = (X^T X)^{-1} X^T y

依赖：statsmodels, numpy
"""

import numpy as np
import statsmodels.api as sm

def run_example(n=100):
    np.random.seed(0)
    X = np.random.randn(n, 2)
    beta = np.array([1.5, -2.0])
    y = X.dot(beta) + np.random.randn(n) * 0.5
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    print(model.summary())

if __name__ == '__main__':
    run_example()
