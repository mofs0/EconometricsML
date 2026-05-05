# 4. Transformer

理论概述：基于自注意力机制（self-attention），能并行处理序列并建模长距离依赖。

数学公式（缩放点积注意力）：
$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.$$ 

Python 简要示例：

```python
from torch.nn import Transformer
model = Transformer(d_model=512, nhead=8)
```

中文论文示例：刘四，《Transformer 在金融文本情感分析中的应用》，2021。
