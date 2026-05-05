"""
Ridge & Lasso 示例（中英）

Usage: python examples/ml/02_RidgeLasso.py
依赖：scikit-learn, numpy
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso

def run_example():
    np.random.seed(1)
    X = np.random.randn(100,10)
    beta = np.zeros(10)
    beta[:3] = [1.0, -0.5, 0.8]
    y = X.dot(beta) + np.random.randn(100) * 0.2
    r = Ridge(alpha=1.0).fit(X,y)
    l = Lasso(alpha=0.1).fit(X,y)
    print('Ridge coeffs:', r.coef_)
    print('Lasso coeffs:', l.coef_)

if __name__ == '__main__':
    run_example()
