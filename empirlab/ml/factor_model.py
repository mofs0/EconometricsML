"""因子模型 / PCA 模板。"""


class FactorModel:
    """因子模型模板。"""

    def __init__(self):
        self.is_fitted = False

    def fit_pca(self, X):
        self.is_fitted = True
        return self

    def fit_fa(self, X):
        self.is_fitted = True
        return self

    def factor_loadings(self):
        return None
