"""
Q-learning 示例（中英）

说明：离散状态简单 Q-learning 伪代码示例，不依赖 gym。
Usage: python examples/rl/01_Q_learning.py
依赖：numpy
"""

import numpy as np
from collections import defaultdict

def run_example():
    n_states = 10
    n_actions = 2
    Q = defaultdict(lambda: np.zeros(n_actions))
    alpha = 0.1
    gamma = 0.9
    for ep in range(50):
        s = np.random.randint(n_states)
        for t in range(20):
            a = np.random.randint(n_actions)
            s2 = np.random.randint(n_states)
            r = np.random.randn()
            Q[s][a] += alpha * (r + gamma * Q[s2].max() - Q[s][a])
            s = s2
    print('Q-learning example finished. sample Q[0]:', Q[0])

if __name__ == '__main__':
    run_example()
