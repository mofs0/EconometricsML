# 3. 决策树与随机森林（Decision Tree / Random Forest）

理论概述：决策树基于特征的分裂构建叶节点，随机森林通过多棵树的装袋（bagging）提高鲁棒性与泛化。

数学公式（信息增益示意）：
$$IG=H(parent)-\sum_{k}\frac{N_k}{N}H(child_k).$$

Python 示例：

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100).fit(X,y)
```

中文论文示例：何三，《随机森林在信用评分中的应用》，2018。
