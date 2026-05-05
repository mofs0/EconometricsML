# 6. 广义矩估计（GMM）

理论概述：GMM 利用矩条件 $E[g(Z,\theta)]=0$ 进行估计，适用于异方差、自相关或内生性情形的稳健估计。

数学公式：GMM 估计量
$$\hat{\theta}=\operatorname{argmin}_\theta \bar{g}_n(\theta)'W_n\bar{g}_n(\theta),$$
其中 $\bar{g}_n(\theta)=\frac{1}{n}\sum_{i}g_i(\theta)$。

Python 简要示例：

```python
from statsmodels.sandbox.regression.gmm import GMM
# 需要自定义 moment 函数，示例略
```

中文论文示例：李九，《面板数据的GMM估计与货币传导研究》，2016。
