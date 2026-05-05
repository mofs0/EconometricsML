# 2. 卷积神经网络（CNN）

理论概述：CNN 使用卷积层提取局部特征，适合图像与一维序列（如金融时间序列的局部模式）。

数学公式（卷积）：一维卷积 $ (x*w)[t]=\sum_{k}x[t-k]w[k]$。

Python 示例（PyTorch）：

```python
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1,16,3)
    def forward(self,x):
        return self.conv(x)
```

中文论文示例：孙二，《CNN 在金融异常检测中的应用》，2020。
