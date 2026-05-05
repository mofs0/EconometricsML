"""
Transformer 示例（中英）

说明：使用 PyTorch 的 Transformer 模块示例。
Usage: python examples/deep_learning/04_Transformer.py
依赖：torch
"""

import torch
from torch.nn import Transformer

def run_example():
    model = Transformer(d_model=32, nhead=4)
    src = torch.randn(10,2,32)  # seq_len, batch, d_model
    tgt = torch.randn(5,2,32)
    out = model(src, tgt)
    print('Transformer output shape:', out.shape)

if __name__ == '__main__':
    run_example()
