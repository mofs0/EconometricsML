"""
SVM / KNN / Naive Bayes 示例（中英）

Usage: python examples/ml/05_SVM_KNN_NB.py
依赖：scikit-learn, numpy
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification

def run_example():
    X,y = make_classification(n_samples=200, n_features=6, random_state=4)
    SVC().fit(X,y)
    KNeighborsClassifier().fit(X,y)
    GaussianNB().fit(X,y)
    print('SVM/KNN/NB trained on synthetic data')

if __name__ == '__main__':
    run_example()
