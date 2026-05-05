"""
Autoencoder 示例（中英）

Usage: python examples/deep_learning/05_Autoencoder.py
依赖：torch
"""

import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, in_dim=100, hid=16):
        super().__init__()
        self.enc = nn.Linear(in_dim, hid)
        self.dec = nn.Linear(hid, in_dim)
    def forward(self,x):
        z = torch.relu(self.enc(x))
        return self.dec(z)

def run_example():
    x = torch.randn(8,100)
    model = AE()
    out = model(x)
    print('AE output shape:', out.shape)

if __name__ == '__main__':
    run_example()
