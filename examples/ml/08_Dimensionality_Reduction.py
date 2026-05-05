"""
PCA / t-SNE 降维示例（中英）

Usage: python examples/ml/08_Dimensionality_Reduction.py
依赖：scikit-learn, numpy
"""

import numpy as np
from sklearn.decomposition import PCA

def run_example():
    np.random.seed(7)
    X = np.random.randn(100,20)
    X2 = PCA(n_components=2).fit_transform(X)
    print('PCA shape:', X2.shape)

if __name__ == '__main__':
    run_example()
