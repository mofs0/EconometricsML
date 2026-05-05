# 10. 二项选择模型：Probit / Logit

理论概述：用于二元结果的概率建模，Logit 使用逻辑函数，Probit 使用标准正态分布函数。

数学公式：Logit
$$P(y=1|x)=\frac{1}{1+e^{-x^T\beta}}. $$
Probit
$$P(y=1|x)=\Phi(x^T\beta),$$
其中 $\Phi$ 为标准正态分布函数。

Python 简要示例：

```python
import statsmodels.api as sm
res = sm.Logit(y, sm.add_constant(X)).fit()
print(res.summary())
```

中文论文示例：胡十三，《Logit 模型在信贷违约预测中的应用》，2012。
