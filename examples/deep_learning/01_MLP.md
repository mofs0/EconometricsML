# 1. 多层感知机（MLP）

理论概述：前馈神经网络，由若干全连接层和非线性激活函数组成，常用于回归与分类任务。

数学公式：单层前向传播 $h=\sigma(Wx+b)$，多层可递归堆叠。

Python 示例（PyTorch）：

```python
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(), nn.Linear(hid,out_dim))
    def forward(self,x):
        return self.net(x)
```

中文论文示例：徐一，《深度多层感知机在风险预测中的应用》，2021。
