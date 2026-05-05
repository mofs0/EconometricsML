"""
SARSA 示例（中英）

说明：on-policy 值更新的 SARSA 伪代码。
Usage: python examples/rl/02_SARSA.py
依赖：numpy
"""

import numpy as np
from collections import defaultdict

def run_example():
    n_states = 8
    n_actions = 2
    Q = defaultdict(lambda: np.zeros(n_actions))
    alpha = 0.1
    gamma = 0.95
    for ep in range(40):
        s = np.random.randint(n_states)
        a = np.random.randint(n_actions)
        for t in range(15):
            s2 = np.random.randint(n_states)
            a2 = np.random.randint(n_actions)
            r = np.random.randn()
            Q[s][a] += alpha * (r + gamma * Q[s2][a2] - Q[s][a])
            s, a = s2, a2
    print('SARSA example finished. sample Q[0]:', Q[0])

if __name__ == '__main__':
    run_example()
