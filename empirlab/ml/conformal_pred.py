"""符合预测区间模板。"""


class ConformalPred:
    """Conformal prediction 模型模板。"""

    def calibrate(self, X_cal, y_cal):
        return self

    def predict_interval(self, X):
        return None

    def coverage_test(self):
        return None
