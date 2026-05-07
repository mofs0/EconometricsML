"""因果森林模板。"""


class CausalForest:
    """因果森林模板。"""

    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, T):
        """拟合因果森林。"""
        self.is_fitted = True
        return self

    def cate(self, X):
        """条件平均处理效应占位。"""
        return None

    def best_linear_proj(self):
        """线性投影占位。"""
        return None


if __name__ == "__main__":
    print("CausalForest 占位")
