# 4. 梯度提升与 XGBoost

理论概述：Boosting 逐步拟合残差，梯度提升树使用梯度下降思想迭代优化损失函数。

数学公式（目标更新示意）：若损失为 $L$, 每步拟合负梯度 $-g_i= -[\partial L(y_i,f(x_i))/\partial f(x_i)]$。

Python 示例：

```python
import xgboost as xgb
model = xgb.XGBClassifier().fit(X,y)
```

中文论文示例：马四，《XGBoost 在股票方向预测中的应用》，2021。
