# 8. 降维：PCA / t-SNE

理论概述：PCA 通过最大化方差或最小化重构误差求主成分，t-SNE 用于非线性低维可视化。

数学公式（PCA）：求解协方差矩阵特征向量 $C=\frac{1}{n}X^TX$，主成分为前 k 个特征向量。

Python 示例：

```python
from sklearn.decomposition import PCA
PCA(n_components=2).fit_transform(X)
```

中文论文示例：吕八，《PCA 在因子提取中的实证分析》，2016。
