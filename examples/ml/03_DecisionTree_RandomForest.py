"""
Decision Tree & Random Forest 示例（中英）

Usage: python examples/ml/03_DecisionTree_RandomForest.py
依赖：scikit-learn, numpy
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def run_example():
    X,y = make_classification(n_samples=200, n_features=5, random_state=2)
    rf = RandomForestClassifier(n_estimators=50, random_state=2).fit(X,y)
    print('feature importances:', rf.feature_importances_)

if __name__ == '__main__':
    run_example()
