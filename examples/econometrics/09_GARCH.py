"""
GARCH 示例（中英）

说明：演示使用 arch 库拟合 GARCH(1,1)。
Usage: python examples/econometrics/09_GARCH.py

依赖：arch, numpy
"""

import numpy as np
from arch import arch_model

def run_example():
    np.random.seed(7)
    y = np.random.randn(1000)
    am = arch_model(y, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    print(res.summary())

if __name__ == '__main__':
    run_example()
