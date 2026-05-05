"""
Linear Regression 示例（中英）

Usage: python examples/ml/01_LinearRegression.py
依赖：scikit-learn, numpy
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def run_example():
    np.random.seed(0)
    X = np.random.randn(100,3)
    beta = np.array([0.5, -1.0, 2.0])
    y = X.dot(beta) + np.random.randn(100) * 0.3
    model = LinearRegression().fit(X,y)
    print('coeffs:', model.coef_)

if __name__ == '__main__':
    run_example()
