# 1. 线性回归（Linear Regression）

理论概述：建模因变量与自变量之间的线性关系，最小二乘估计。

数学公式：$y=X\beta+\varepsilon,\;\hat{\beta}=(X^TX)^{-1}X^Ty$。

Python 示例：

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
print(model.coef_)
```

中文论文示例：陈一，《基于线性回归的宏观变量预测研究》，2019。
