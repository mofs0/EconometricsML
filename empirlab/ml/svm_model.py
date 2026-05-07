"""SVM 模板。"""


class SVMModel:
    """支持向量机模板。"""

    def __init__(self):
        self.is_fitted = False

    def fit_svc(self, X, y):
        """分类拟合。"""
        self.is_fitted = True
        return self

    def fit_svr(self, X, y):
        """回归拟合。"""
        self.is_fitted = True
        return self

    def kernel_selection(self):
        """核函数选择占位。"""
        return "rbf"


if __name__ == "__main__":
    print(SVMModel().kernel_selection())
