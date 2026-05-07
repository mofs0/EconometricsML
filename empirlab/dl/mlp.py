"""PyTorch MLP 最小实现（仅作为模板）。

注意：项目约定只使用 PyTorch，不使用 TensorFlow/Keras。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MLP(nn.Module):
    """简单的多层感知机，用于回归任务的模板实现。

    参数注释见类 docstring，底部包含训练的最小示例。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_example():
    # 最小训练示例，生成合成数据并做几步 SGD
    torch.manual_seed(0)
    X = torch.randn(128, 3)
    y = X[:, 0] * 1.2 + X[:, 1] * -0.7 + torch.randn(128) * 0.1

    model = MLP(input_dim=3)
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(20):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    print("训练完成，最后 loss：", loss.item())


if __name__ == "__main__":
    train_example()
