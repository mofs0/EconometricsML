# 10. 模型评估与交叉验证

理论概述：模型评估包括回归与分类指标、交叉验证用于估计泛化误差。

常见指标：MSE, MAE, ROC-AUC, Accuracy 等。

Python 示例：

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
```

中文论文示例：张十，《交叉验证在资产定价模型比较中的作用》，2019。
