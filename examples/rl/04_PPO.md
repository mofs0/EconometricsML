# 4. 近端策略优化（PPO）

理论概述：PPO 使用裁剪的策略梯度目标或 KL 约束来稳定策略更新，是当前常用的策略优化算法之一。

数学公式（裁剪目标）：
$$L^{CLIP}(\theta)=E\left[\min\left(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right],$$
其中 $r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。

中文论文示例：范四，《PPO 在投资组合管理中的应用研究》，2022。
