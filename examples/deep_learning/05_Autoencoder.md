# 5. 自编码器（Autoencoder）

理论概述：自编码器通过编码器-解码器结构学习数据的低维表征，常用于降噪与异常检测。

数学公式：最小化重构误差
$$\min_{\theta} \sum_i ||x_i-\hat{x}_i||^2,\quad \hat{x}=g_\theta(f_\theta(x)).$$

Python 示例（PyTorch）：

```python
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(100, 10)
        self.dec = nn.Linear(10, 100)
    def forward(self,x):
        z = torch.relu(self.enc(x))
        return self.dec(z)
```

中文论文示例：黄五，《自编码器在金融异常检测中的研究》，2020。
