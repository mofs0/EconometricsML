# 7. ARIMA（自回归积分滑动平均）

理论概述：用于时间序列建模，结合自回归 (AR)、差分 (I) 与移动平均 (MA) 部分。

数学公式：ARIMA(p,d,q) 的一般形式
$$\phi(L)(1-L)^d y_t=\theta(L)\varepsilon_t,$$
其中 $\phi(L)=1-\phi_1L-...-\phi_pL^p,\;\theta(L)=1+\theta_1L+...+\theta_qL^q$。

Python 简要示例：

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(y, order=(1,1,1)).fit()
print(model.summary())
```

中文论文示例：周十，《基于ARIMA模型的宏观经济指标预测》，2015。
