# 1. Q-learning

理论概述：基于值的无模型强化学习，学习状态-动作价值函数 $Q(s,a)$，使用贝尔曼最优方程更新。

数学公式：Q-learning 更新
$$Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma\max_a Q(s_{t+1},a)-Q(s_t,a_t)\right].$$

Python 简要示例：

```python
# 离散环境的简单 Q-learning 伪代码
Q = defaultdict(lambda: np.zeros(n_actions))
for ep in range(episodes):
    s = env.reset()
    while not done:
        a = choose_action(Q[s])
        s2, r, done, _ = env.step(a)
        Q[s][a] += alpha*(r + gamma*max(Q[s2]) - Q[s][a])
        s = s2
```

中文论文示例：方一，《Q-learning 在资产配置动态策略中的应用》，2019。
