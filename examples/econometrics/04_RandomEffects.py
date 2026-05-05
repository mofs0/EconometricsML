"""
Random Effects 示例（中英双语注释）

说明：使用 RandomEffects 估计面板随机效应模型。
Usage: python examples/econometrics/04_RandomEffects.py
依赖：linearmodels, numpy, pandas
"""

import numpy as np
import pandas as pd
from linearmodels.panel import RandomEffects

def run_example():
    np.random.seed(3)
    n_entities = 30
    n_time = 4
    rows = []
    for i in range(n_entities):
        alpha = np.random.randn() * 0.5
        for t in range(n_time):
            x = np.random.randn()
            y = 0.7 * x + alpha + np.random.randn() * 0.2
            rows.append({'entity':f'e{i}','time':t,'y':y,'x':x})
    df = pd.DataFrame(rows).set_index(['entity','time'])
    mod = RandomEffects(df['y'], df[['x']]).fit()
    print(mod.summary)

if __name__ == '__main__':
    run_example()
