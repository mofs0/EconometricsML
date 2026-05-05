# 2. 正则化：Ridge 与 Lasso

理论概述：通过惩罚项减少过拟合，Ridge 使用 L2 惩罚，Lasso 使用 L1 惩罚并可做变量选择。

数学公式：Ridge
$$\hat{\beta}=\operatorname{argmin}_\beta ||y-X\beta||_2^2+\lambda||\beta||_2^2.$$ 
Lasso
$$\hat{\beta}=\operatorname{argmin}_\beta ||y-X\beta||_2^2+\lambda||\beta||_1.$$ 

Python 示例：

```python
from sklearn.linear_model import Ridge, Lasso
Ridge().fit(X,y)
Lasso().fit(X,y)
```

中文论文示例：孙二，《Lasso 在高维资产定价中的应用》，2020。
