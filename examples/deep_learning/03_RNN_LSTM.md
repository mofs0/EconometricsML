# 3. RNN / LSTM

理论概述：循环神经网络用于序列建模，LSTM 通过门控机制缓解长期依赖问题。

数学公式（LSTM 单元简要）：
$$f_t=\sigma(W_f[x_t,h_{t-1}]+b_f),\quad i_t=\sigma(W_i[...])$$
$$\tilde{c}_t=\tanh(W_c[...]),\quad c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t$$

Python 示例（PyTorch）：

```python
import torch.nn as nn
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1)
```

中文论文示例：周三，《LSTM 在高频序列预测中的效果研究》，2019。
