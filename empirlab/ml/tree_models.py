"""树模型模板（Random Forest / Boosting）。"""


class TreeModels:
    """树模型模板。"""

    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        """拟合树模型。"""
        self.is_fitted = True
        return self

    def feature_importance(self):
        """特征重要性占位。"""
        return None

    def summary(self):
        return {"note": "TreeModels 占位实现"}


if __name__ == "__main__":
    m = TreeModels()
    m.fit(None, None)
    print(m.summary())
