# 3. 深度 Q 网络（DQN）

理论概述：结合深度神经网络近似 Q 函数，并使用经验回放与目标网络稳定训练。

数学核心：最小化均方贝尔曼误差
$$L(\theta)=E_{(s,a,r,s')}\left[\left(r+\gamma\max_{a'}Q(s',a';\theta^-)-Q(s,a;\theta)\right)^2\right].$$

Python 简要示例：可参考 OpenAI baselines / stable-baselines3 的 DQN 实现。

中文论文示例：郭三，《DQN 在高频交易策略构建中的探索》，2021。
