"""投资组合交易环境（Gymnasium 接口）
=====================================
来源参考：Mnih, V., et al. (2015). Human-level control through deep
          reinforcement learning. *Nature*, 518, 529–533.
          Jiang, Z., Xu, D., & Liang, J. (2017). A deep reinforcement
          learning framework for the financial portfolio management problem.
          arXiv:1706.10059.

环境说明：
  - 状态 (state)：过去 window 步的 n 只资产收益率矩阵，shape = (window, n_assets)
  - 动作 (action)：离散化的持仓比例（0=全现金, 1=均等持有, 2=集中持有最大权重）
    或连续动作（n_assets 维权重向量，需配合 PPO/DDPG）
  - 奖励 (reward)：当步对数收益率
  - 终止 (done)：数据用完

使用方式：
  env = PortfolioEnv(returns_df)   # returns_df: (T, n_assets) 收益率 DataFrame
  obs, info = env.reset()
  obs, reward, done, trunc, info = env.step(action)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


class PortfolioEnv(gym.Env if _GYM_AVAILABLE else object):
    """简单投资组合强化学习环境。

    参数
    ----
    returns       : pd.DataFrame or np.ndarray, shape (T, n_assets)
                    每步的资产收益率（已去除1，即 r_t = P_t/P_{t-1} - 1）
    window        : int, default 20  —— 观察窗口长度
    transaction_cost : float, default 0.001  —— 单边交易成本
    n_discrete    : int, default 3   —— 离散动作数（均等/集中/全现金）

    示例
    ----
    >>> import numpy as np
    >>> from empirlab.rl import PortfolioEnv
    >>> ret = np.random.randn(500, 3) * 0.01  # 3 只资产，500 步
    >>> env = PortfolioEnv(ret, window=10)
    >>> obs, _ = env.reset()
    >>> obs2, r, done, trunc, info = env.step(0)
    >>> print(f"obs shape={obs2.shape}, reward={r:.4f}")
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns,
        window: int = 20,
        transaction_cost: float = 0.001,
        n_discrete: int = 3,
    ):
        if not _GYM_AVAILABLE:
            raise ImportError("需要安装 gymnasium：pip install gymnasium")
        super().__init__()

        self.returns = (
            returns.to_numpy(dtype=float)
            if isinstance(returns, pd.DataFrame)
            else np.asarray(returns, dtype=float)
        )
        self.T, self.n_assets = self.returns.shape
        self.window = window
        self.transaction_cost = transaction_cost
        self.n_discrete = n_discrete

        # 离散动作空间：0=全现金, 1=均等持有, 2=集中持有最高预期收益资产
        self.action_space = spaces.Discrete(n_discrete)

        # 观察：过去 window 步的收益率矩阵（展平）
        obs_dim = window * self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._t: int = window
        self._weights: np.ndarray = np.ones(self.n_assets) / self.n_assets
        self._portfolio_value: float = 1.0

    def _get_obs(self) -> np.ndarray:
        return self.returns[self._t - self.window: self._t].astype(np.float32).ravel()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.window
        self._weights = np.ones(self.n_assets) / self.n_assets
        self._portfolio_value = 1.0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步交易。

        返回
        ----
        obs, reward, terminated, truncated, info
        """
        # 根据动作更新权重
        new_weights = self._action_to_weights(int(action))

        # 交易成本
        turnover = np.sum(np.abs(new_weights - self._weights))
        cost = turnover * self.transaction_cost

        # 当步收益率
        r_t = self.returns[self._t]
        port_return = float(np.dot(new_weights, r_t)) - cost

        # 更新
        self._weights = new_weights * (1 + r_t)
        w_sum = self._weights.sum()
        self._weights = self._weights / w_sum if w_sum > 0 else new_weights
        self._portfolio_value *= (1 + port_return)
        self._t += 1

        done = self._t >= self.T
        obs = self._get_obs() if not done else np.zeros(self.window * self.n_assets, dtype=np.float32)

        reward = np.log(1 + port_return) if port_return > -1 else -1.0

        return obs, reward, done, False, {
            "portfolio_value": self._portfolio_value,
            "port_return": port_return,
            "weights": self._weights.copy(),
        }

    def _action_to_weights(self, action: int) -> np.ndarray:
        """将离散动作转换为权重向量。"""
        if action == 0:
            # 全现金（实际上均等持有，因无现金资产）
            return np.ones(self.n_assets) / self.n_assets
        elif action == 1:
            # 均等持有
            return np.ones(self.n_assets) / self.n_assets
        elif action == 2:
            # 集中持有过去窗口收益最高的资产
            mean_ret = self.returns[self._t - self.window: self._t].mean(0)
            best = int(np.argmax(mean_ret))
            w = np.zeros(self.n_assets)
            w[best] = 1.0
            return w
        else:
            return np.ones(self.n_assets) / self.n_assets


def make_synthetic_returns(
    n_steps: int = 500,
    n_assets: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成收益率数据，方便测试环境。"""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.0005, 0.002, n_assets)
    sigma = rng.uniform(0.01, 0.02, n_assets)
    returns = rng.normal(mu, sigma, (n_steps, n_assets))
    cols = [f"asset_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=cols)


if __name__ == "__main__":
    if not _GYM_AVAILABLE:
        print("gymnasium 未安装，跳过示例。")
    else:
        ret_df = make_synthetic_returns(n_steps=300, n_assets=3)
        env = PortfolioEnv(ret_df, window=10)
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, done, _, info = env.step(action)
            total_reward += r
        print(f"随机策略总对数收益：{total_reward:.4f}")
        print(f"最终组合价值：{info['portfolio_value']:.4f}")
