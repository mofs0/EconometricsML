# 7. 无监督学习：聚类（KMeans / DBSCAN）

理论概述：KMeans 通过最小化簇内平方和进行划分；DBSCAN 基于密度发现任意形状簇。

数学公式（KMeans 目标）：
$$\min_{C}\sum_{k}\sum_{x_i\in C_k}||x_i-\mu_k||^2.$$ 

Python 示例：

```python
from sklearn.cluster import KMeans, DBSCAN
KMeans(n_clusters=3).fit(X)
DBSCAN(eps=0.5).fit(X)
```

中文论文示例：钱七，《聚类算法在行业分类中的应用比较》，2017。
