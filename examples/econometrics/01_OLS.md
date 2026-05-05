# 1. 普通最小二乘（OLS）

理论概述：OLS 用于线性回归，假定误差项零均值、同方差且与解释变量不相关。估计量通过最小化残差平方和得到。

数学公式：设 $y=X\beta+\varepsilon$，OLS 解为
$$\hat{\beta}=(X^TX)^{-1}X^Ty.$$ 

Python 简要示例：

```python
import statsmodels.api as sm
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

中文论文示例：张三、李四，《基于OLS的高频交易收益分析》，2020。
