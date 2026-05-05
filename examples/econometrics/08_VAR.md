# 8. 向量自回归（VAR）

理论概述：VAR 同时建模多个时间序列变量，捕捉变量间的相互影响与冲击传播。

数学公式：VAR(p)
$$y_t=c+\sum_{i=1}^pA_i y_{t-i}+\varepsilon_t,$$
其中 $y_t$ 为向量，$A_i$ 为系数矩阵。

Python 简要示例：

```python
from statsmodels.tsa.api import VAR
model = VAR(data)
res = model.fit(1)
print(res.summary())
```

中文论文示例：吴十一，《VAR 模型及其在金融溢出效应分析中的应用》，2014。
