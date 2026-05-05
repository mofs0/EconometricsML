"""
Additional machine learning models for the core package.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _as_2d(x):
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class RidgeRegressor:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class LassoRegressor:
    def __init__(self, alpha=1.0):
        self.model = Lasso(alpha=alpha, max_iter=5000)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class RandomForestRegressorModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class GradientBoostingRegressorModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class DecisionTreeRegressorModel:
    def __init__(self, random_state=42):
        self.model = DecisionTreeRegressor(random_state=random_state)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class SVRModel:
    def __init__(self, C=1.0, epsilon=0.1):
        self.model = SVR(C=C, epsilon=epsilon)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class LogisticClassifier:
    def __init__(self, max_iter=200):
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))

    def predict_proba(self, X):
        return self.model.predict_proba(_as_2d(X))[:, 1]


class RandomForestClassifierModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class GradientBoostingClassifierModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class DecisionTreeClassifierModel:
    def __init__(self, random_state=42):
        self.model = DecisionTreeClassifier(random_state=random_state)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class KMeansClusterer:
    def __init__(self, n_clusters=3, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    def fit(self, X):
        self.model.fit(_as_2d(X))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class PCAProjector:
    def __init__(self, n_components=2):
        self.model = PCA(n_components=n_components)

    def fit(self, X):
        self.model.fit(_as_2d(X))
        return self

    def transform(self, X):
        return self.model.transform(_as_2d(X))

    def inverse_transform(self, X):
        return self.model.inverse_transform(_as_2d(X))


try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None
    XGBClassifier = None


class XGBoostRegressorModel:
    def __init__(self, **kwargs):
        if XGBRegressor is None:
            self.model = None
            self._missing = True
        else:
            self.model = XGBRegressor(**kwargs)
            self._missing = False

    def fit(self, X, y):
        if self._missing:
            raise ImportError("xgboost is not installed")
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        if self._missing:
            raise ImportError("xgboost is not installed")
        return self.model.predict(_as_2d(X))


class KNNRegressorModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class KNNClassifierModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


__all__ = [
    "RidgeRegressor",
    "LassoRegressor",
    "RandomForestRegressorModel",
    "GradientBoostingRegressorModel",
    "DecisionTreeRegressorModel",
    "SVRModel",
    "LogisticClassifier",
    "RandomForestClassifierModel",
    "GradientBoostingClassifierModel",
    "DecisionTreeClassifierModel",
    "KMeansClusterer",
    "PCAProjector",
    "XGBoostRegressorModel",
    "KNNRegressorModel",
    "KNNClassifierModel",
]
