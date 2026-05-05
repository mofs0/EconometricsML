# 5. 差分中的差分（Difference-in-Differences, DiD）

理论概述：用于政策评估，通过比较处理组与对照组在处理前后的差异来识别处理效应。

数学公式：基本 DiD 估计 $y_{it}=\alpha+\tau D_{it}+\gamma_t+\delta_i+\varepsilon_{it}$，其中 $\tau$ 为处理效应。

Python 简要示例：

```python
import statsmodels.formula.api as smf
res = smf.ols('y ~ treated * post + C(year) + C(id)', data=df).fit()
print(res.summary())
```

中文论文示例：孙八，《某项财政政策的准实验评估：DiD 方法》，2021。
