"""
Fixed Effects 面板模型示例（中英）

说明：演示如何使用 PanelOLS 进行个体固定效应估计。
Usage: python examples/econometrics/03_FixedEffects.py

依赖：linearmodels, pandas, numpy
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

def run_example():
    np.random.seed(2)
    n_entities = 20
    n_time = 5
    rows = []
    for i in range(n_entities):
        alpha = np.random.randn()  # 个体不变效应
        for t in range(n_time):
            x = np.random.randn()
            y = 1.2 * x + alpha + 0.5 * np.random.randn()
            rows.append({'entity':f'e{i}','time':t,'y':y,'x':x})
    df = pd.DataFrame(rows).set_index(['entity','time'])
    mod = PanelOLS(df['y'], sm_add_const := __import__('statsmodels.api').add_constant(df[['x']]), entity_effects=True).fit()
    print(mod.summary)

if __name__ == '__main__':
    run_example()
