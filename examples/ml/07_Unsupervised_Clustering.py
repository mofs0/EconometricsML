"""
Clustering 示例（KMeans / DBSCAN）中英注释

Usage: python examples/ml/07_Unsupervised_Clustering.py
依赖：scikit-learn, numpy
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs

def run_example():
    X,_ = make_blobs(n_samples=200, centers=3, random_state=6)
    k = KMeans(n_clusters=3, random_state=6).fit(X)
    db = DBSCAN(eps=0.5).fit(X)
    print('KMeans labels sample:', k.labels_[:10])

if __name__ == '__main__':
    run_example()
