"""
Gradient Boosting / XGBoost 示例（中英）

Usage: python examples/ml/04_GradientBoosting_XGBoost.py
依赖：xgboost, sklearn, numpy
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

def run_example():
    X,y = make_classification(n_samples=300, n_features=10, random_state=3)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X,y)
    print('trained XGBoost, n_classes:', len(np.unique(y)))

if __name__ == '__main__':
    run_example()
