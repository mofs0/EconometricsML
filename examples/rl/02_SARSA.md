# 2. SARSA

理论概述：SARSA 为 on-policy 值更新方法，使用当前策略生成的后续动作参与更新，区别于 off-policy 的 Q-learning。

数学公式：
$$Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\right].$$

简要示例：与 Q-learning 相似，但使用下一步实际选择的动作更新。

中文论文示例：秦二，《SARSA 在算法交易策略学习中的探索》，2020。
