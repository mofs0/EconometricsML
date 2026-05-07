"""BART 模板。"""


class BARTModel:
    """贝叶斯回归树模板。"""

    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = True
        return self

    def predict(self, X):
        return None

    def credible_interval(self):
        return None
