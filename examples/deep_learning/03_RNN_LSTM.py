"""
RNN / LSTM 示例（中英）

Usage: python examples/deep_learning/03_RNN_LSTM.py
依赖：torch
"""

import torch
import torch.nn as nn

def run_example():
    lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1)
    x = torch.randn(5,4,8)  # seq_len, batch, input_size
    out, (h,c) = lstm(x)
    print('LSTM output shape:', out.shape)

if __name__ == '__main__':
    run_example()
