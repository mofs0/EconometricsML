# 5. SVM / KNN / 朴素贝叶斯

理论概述：
- SVM 寻找最大间隔超平面；
- KNN 基于最近邻的非参数方法；
- 朴素贝叶斯基于特征条件独立性假设进行概率预测。

数学要点（SVM 硬间隔）：
$$\min_{w} \frac{1}{2}||w||^2\quad s.t.\; y_i(w^Tx_i+b)\ge1.$$ 

Python 示例：

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
SVC().fit(X,y)
KNeighborsClassifier().fit(X,y)
GaussianNB().fit(X,y)
```

中文论文示例：林五，《SVM 在高频交易信号分类中的比较研究》，2019。
