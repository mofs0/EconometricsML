"""Deep Q-Network (DQN) 代理
==============================
来源参考：Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).
          Human-level control through deep reinforcement learning.
          *Nature*, 518(7540), 529–533.

核心技术：
  - Experience Replay（经验回放缓冲区）
  - Target Network（目标网络，每 C 步同步）
  - ε-贪婪策略（线性衰减）

适配：PortfolioEnv（离散动作空间）
"""
from __future__ import annotations

import sys
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _QNet(nn.Module if _TORCH_AVAILABLE else object):
    """三层 MLP Q 函数。"""

    def __init__(self, obs_dim, n_actions, hidden=128):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区。"""

    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(act, dtype=np.int64),
            np.array(rew, dtype=np.float32),
            np.array(nobs, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """DQN 代理。

    参数
    ----
    obs_dim      : int   —— 观察维度
    n_actions    : int   —— 离散动作数
    hidden       : int, default 128
    lr           : float, default 1e-3
    gamma        : float, default 0.99  —— 折扣因子
    eps_start    : float, default 1.0
    eps_end      : float, default 0.05
    eps_decay    : int, default 500     —— 衰减步数
    batch_size   : int, default 64
    buffer_size  : int, default 10_000
    target_update: int, default 100     —— 目标网络同步步数
    device       : str, default "auto"

    示例
    ----
    >>> from empirlab.rl import PortfolioEnv, DQNAgent
    >>> from empirlab.rl.portfolio_env import make_synthetic_returns
    >>> ret = make_synthetic_returns(300, 3)
    >>> env = PortfolioEnv(ret, window=10)
    >>> agent = DQNAgent(obs_dim=30, n_actions=3, eps_decay=200)
    >>> rewards = agent.train(env, n_episodes=5)
    >>> print(f"最后一轮总奖励：{rewards[-1]:.4f}")
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 500,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        target_update: int = 100,
        device: str = "auto",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch：pip install torch")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.policy_net = _QNet(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = _QNet(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self._steps = 0

    @property
    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * max(
            0.0, 1 - self._steps / self.eps_decay
        )

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        obs_t = torch.tensor(obs[None], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(obs_t).argmax(dim=1).item())

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)

        obs_t  = torch.tensor(obs).to(self.device)
        act_t  = torch.tensor(act).to(self.device)
        rew_t  = torch.tensor(rew).to(self.device)
        nobs_t = torch.tensor(nobs).to(self.device)
        done_t = torch.tensor(done).to(self.device)

        q_curr = self.policy_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(nobs_t).max(1).values
            q_target = rew_t + self.gamma * q_next * (1 - done_t)

        loss = nn.SmoothL1Loss()(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def train(self, env, n_episodes: int = 50, verbose: bool = False) -> list:
        """训练 n_episodes 轮并返回每轮总奖励。"""
        rewards = []
        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            total_r = 0.0
            done = False
            while not done:
                action = self.select_action(obs)
                nobs, r, done, _, _ = env.step(action)
                self.buffer.push(obs, action, r, nobs, float(done))
                self.update()
                obs = nobs
                total_r += r
            rewards.append(total_r)
            if verbose and ep % 10 == 0:
                print(f"  Episode {ep:3d}/{n_episodes}  "
                      f"total_r={total_r:.4f}  eps={self.epsilon:.3f}")
        return rewards


if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        print("PyTorch 未安装，跳过示例。")
    else:
        try:
            from empirlab.rl.portfolio_env import PortfolioEnv, make_synthetic_returns
        except ImportError:
            from portfolio_env import PortfolioEnv, make_synthetic_returns

        ret = make_synthetic_returns(500, 3)
        env = PortfolioEnv(ret, window=10)
        agent = DQNAgent(obs_dim=30, n_actions=3, eps_decay=300)
        rewards = agent.train(env, n_episodes=20, verbose=True)
        print(f"\n最后5轮平均奖励：{np.mean(rewards[-5:]):.4f}")
