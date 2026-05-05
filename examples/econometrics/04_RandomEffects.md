# 4. 随机效应模型（Random Effects）

理论概述：面板数据中假定个体效应为随机且与解释变量不相关，可用广义最小二乘（GLS）估计。

数学公式：模型 $y_{it}=X_{it}\beta+u_{it}$，其中 $u_{it}=\alpha_i+\varepsilon_{it}$，RE 估计使用变换矩阵 $\Omega$。

Python 简要示例：

```python
from linearmodels.panel import RandomEffects
mod = RandomEffects(y, X).fit()
print(mod.summary)
```

中文论文示例：钱七，《随机效应在区域增长研究中的应用》，2017。
