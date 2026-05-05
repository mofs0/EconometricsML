"""
Additional reinforcement learning agents for the core package.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDRegressor


def _state_key(state):
    arr = np.asarray(state, dtype=float).reshape(-1)
    return tuple(np.round(arr, 2))


def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state, training=True):
        key = _state_key(state)
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[key]))

    def learn(self, state, action, reward, next_state, done=False):
        key = _state_key(state)
        next_key = _state_key(next_state)
        target = reward + (0.0 if done else self.discount_factor * np.max(self.q_table[next_key]))
        self.q_table[key][action] += self.learning_rate * (target - self.q_table[key][action])


class SARSAAgent:
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state, training=True):
        key = _state_key(state)
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[key]))

    def learn(self, state, action, reward, next_state, next_action, done=False):
        key = _state_key(state)
        target = reward + (0.0 if done else self.discount_factor * self.q_table[_state_key(next_state)][next_action])
        self.q_table[key][action] += self.learning_rate * (target - self.q_table[key][action])


class DQNAgent:
    """Very small DQN-style agent using one linear SGD regressor per action."""

    def __init__(self, state_dim, n_actions, learning_rate=0.01, discount_factor=0.95):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.models = [SGDRegressor(learning_rate="constant", eta0=learning_rate, random_state=42) for _ in range(n_actions)]
        self._initialized = [False] * n_actions

    def _predict_q(self, state):
        state = np.asarray(state).reshape(1, -1)
        q = []
        for i, model in enumerate(self.models):
            if not self._initialized[i]:
                q.append(0.0)
            else:
                q.append(float(model.predict(state)[0]))
        return np.asarray(q)

    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._predict_q(state)))

    def learn(self, state, action, reward, next_state, done=False):
        state = np.asarray(state).reshape(1, -1)
        next_q = 0.0 if done else float(np.max(self._predict_q(next_state)))
        target = reward + self.discount_factor * next_q
        if not self._initialized[action]:
            self.models[action].partial_fit(state, np.array([target]))
            self._initialized[action] = True
        else:
            self.models[action].partial_fit(state, np.array([target]))


class PPOAgent:
    """Small policy-gradient style agent with clipped updates."""

    def __init__(self, state_dim, n_actions, learning_rate=0.05, clip_ratio=0.2):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.weights = np.zeros((state_dim, n_actions))

    def policy(self, state):
        state = np.asarray(state).reshape(-1)
        logits = state @ self.weights
        return _softmax(logits)

    def choose_action(self, state, training=True):
        probs = self.policy(state)
        if training:
            return int(np.random.choice(self.n_actions, p=probs))
        return int(np.argmax(probs))

    def learn(self, states, actions, advantages):
        states = np.asarray(states)
        actions = np.asarray(actions)
        advantages = np.asarray(advantages)
        for s, a, adv in zip(states, actions, advantages):
            probs = self.policy(s)
            old_prob = probs[a]
            clipped_adv = np.clip(adv, -1.0 / self.clip_ratio, 1.0 / self.clip_ratio)
            grad = np.outer(np.asarray(s).reshape(-1), np.eye(self.n_actions)[a] - probs)
            self.weights += self.learning_rate * clipped_adv * grad


class A2CAgent:
    """Linear actor-critic agent."""

    def __init__(self, state_dim, n_actions, learning_rate=0.05, discount_factor=0.95):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actor_weights = np.zeros((state_dim, n_actions))
        self.critic_weights = np.zeros(state_dim)

    def policy(self, state):
        state = np.asarray(state).reshape(-1)
        return _softmax(state @ self.actor_weights)

    def choose_action(self, state, training=True):
        probs = self.policy(state)
        return int(np.random.choice(self.n_actions, p=probs)) if training else int(np.argmax(probs))

    def value(self, state):
        state = np.asarray(state).reshape(-1)
        return float(state @ self.critic_weights)

    def learn(self, state, action, reward, next_state, done=False):
        state = np.asarray(state).reshape(-1)
        next_value = 0.0 if done else self.value(next_state)
        td_error = reward + self.discount_factor * next_value - self.value(state)

        # critic update
        self.critic_weights += self.learning_rate * td_error * state

        # actor update
        probs = self.policy(state)
        grad = np.outer(state, np.eye(self.n_actions)[action] - probs)
        self.actor_weights += self.learning_rate * td_error * grad


__all__ = [
    "QLearningAgent",
    "SARSAAgent",
    "DQNAgent",
    "PPOAgent",
    "A2CAgent",
]
