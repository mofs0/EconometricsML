# 6. 逻辑回归与分类

理论概述：逻辑回归用于二分类，建模事件概率并使用极大似然估计参数。

数学公式：对数似然为
$$\ell(\beta)=\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)],\;p_i=\frac{1}{1+e^{-x_i^T\beta}}.$$ 

Python 示例：

```python
from sklearn.linear_model import LogisticRegression
LogisticRegression().fit(X,y)
```

中文论文示例：周六，《逻辑回归在违约概率预测中的实证研究》，2018。
