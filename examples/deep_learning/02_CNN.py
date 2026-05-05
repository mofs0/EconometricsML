"""
CNN 示例（中英）

说明：1D 卷积示例，适用于序列数据。
Usage: python examples/deep_learning/02_CNN.py
依赖：torch, numpy
"""

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1,16,3)
    def forward(self,x):
        return self.conv(x)

def run_example():
    x = torch.randn(8,1,50)  # batch, channel, seq_len
    model = SimpleCNN()
    y = model(x)
    print('CNN output shape:', y.shape)

if __name__ == '__main__':
    run_example()
