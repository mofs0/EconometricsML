"""
Q 学习智能体 (Q-Learning Agent)
===============================
来源参考：Sutton and Barto (2018) Reinforcement Learning
适用场景：离散状态-动作空间的策略学习
Python 版本：3.10+
依赖：numpy >= 1.21 / pandas >= 1.3 / gymnasium >= 0.29
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym

from empirlab.utils.metrics import calculate_metrics


class QLearningAgent:
    """基于 Gymnasium 接口的 Q-learning。"""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_fitted = False
        self.q_table: np.ndarray | None = None
        self._episode_rewards: list[float] = []

    def fit(self, env_name: str = "FrozenLake-v1", episodes: int = 300, **kwargs):
        """在离散环境上训练 Q 表。"""
        env = gym.make(env_name, is_slippery=False)
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table = np.zeros((n_states, n_actions), dtype=float)

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                if np.random.rand() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(self.q_table[state]))

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)

                td_target = reward + self.gamma * np.max(self.q_table[next_state])
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error
                state = next_state
                total_reward += reward

            self._episode_rewards.append(total_reward)

        env.close()
        self.is_fitted = True
        return self

    def predict(self, states):
        """给定状态返回贪心动作。"""
        if not self.is_fitted or self.q_table is None:
            raise RuntimeError("请先调用 fit() 方法。")
        states_arr = np.asarray(states, dtype=int).reshape(-1)
        return np.array([int(np.argmax(self.q_table[s])) for s in states_arr])

    def summary(self):
        """返回训练摘要。"""
        if not self.is_fitted or self.q_table is None:
            raise RuntimeError("请先调用 fit() 方法。")
        rewards = np.asarray(self._episode_rewards, dtype=float)
        reward_pred = np.full_like(rewards, rewards.mean())
        metrics = calculate_metrics(rewards, reward_pred)
        return {
            "model": "QLearningAgent",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes": int(rewards.shape[0]),
            "average_reward": float(rewards.mean()),
            "reward_dispersion": metrics,
        }


if __name__ == "__main__":
    np.random.seed(42)
    model = QLearningAgent()
    model.fit(episodes=200)
    result = model.summary()
    print("训练摘要:", pd.Series(result).to_dict())
