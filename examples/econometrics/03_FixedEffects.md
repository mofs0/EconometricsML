# 3. 固定效应模型（Fixed Effects）

理论概述：用于面板数据，消除个体不变的未观测效应。常用两种变换：双向差分或时间固定效应。

数学公式：个体固定效应模型 $y_{it}=X_{it}\beta+\alpha_i+\varepsilon_{it}$，去均值变换后为
$$\tilde{y}_{it}=\tilde{X}_{it}\beta+\tilde{\varepsilon}_{it},$$
其中 $\tilde{y}_{it}=y_{it}-\bar{y}_i$。

Python 简要示例：

```python
from linearmodels.panel import PanelOLS
mod = PanelOLS(y, X, entity_effects=True).fit()
print(mod.summary)
```

中文论文示例：赵六，《面板数据固定效应与行业收益分析》，2019。
