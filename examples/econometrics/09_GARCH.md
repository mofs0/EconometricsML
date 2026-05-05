# 9. GARCH（广义自回归条件异方差）

理论概述：用于刻画金融时间序列的条件异方差（波动聚集）现象，常用 GARCH(p,q) 模型。

数学公式：GARCH(1,1)
$$y_t=\mu+\varepsilon_t,\quad \varepsilon_t=\sigma_t z_t,\quad \sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2.$$ 

Python 简要示例：

```python
from arch import arch_model
am = arch_model(y, vol='Garch', p=1, q=1)
res = am.fit(disp='off')
print(res.summary())
```

中文论文示例：郑十二，《基于GARCH的中国股市波动性研究》，2013。
