"""Ridge / Lasso / ElasticNet 模板。"""


class RegularizedModel:
    """正则化线性模型模板。"""

    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, penalty="ridge"):
        """拟合正则化模型。"""
        self.is_fitted = True
        return self

    def coef_path(self):
        """系数路径占位。"""
        return None

    def summary(self):
        return {"note": "RegularizedModel 占位实现"}


if __name__ == "__main__":
    m = RegularizedModel()
    m.fit(None, None)
    print(m.summary())
