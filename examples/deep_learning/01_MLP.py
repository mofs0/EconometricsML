"""
MLP 示例（中英）

说明：多层感知机示例，使用 PyTorch。
Usage: python examples/deep_learning/01_MLP.py
依赖：torch, numpy
"""

import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim=10, hid=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(), nn.Linear(hid,out_dim))
    def forward(self,x):
        return self.net(x)

def run_example():
    x = torch.randn(16,10)
    model = MLP()
    out = model(x)
    print('MLP output shape:', out.shape)

if __name__ == '__main__':
    run_example()
