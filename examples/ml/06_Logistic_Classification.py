"""
Logistic Regression 示例（中英）

Usage: python examples/ml/06_Logistic_Classification.py
依赖：scikit-learn, numpy
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def run_example():
    X,y = make_classification(n_samples=200, n_features=5, random_state=5)
    model = LogisticRegression(max_iter=200).fit(X,y)
    print('coeffs:', model.coef_)

if __name__ == '__main__':
    run_example()
