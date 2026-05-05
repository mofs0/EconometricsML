"""
模型评估示例（中英）

Usage: python examples/ml/10_ModelEvaluation.py
依赖：scikit-learn, numpy
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def run_example():
    X,y = make_regression(n_samples=200, n_features=5, noise=0.3, random_state=9)
    model = RandomForestRegressor(n_estimators=50, random_state=9)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print('CV MSE (neg):', scores)

if __name__ == '__main__':
    run_example()
