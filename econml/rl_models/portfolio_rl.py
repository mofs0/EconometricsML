"""
Reinforcement Learning Agent for Portfolio Optimization

Uses Q-learning and policy gradient methods for dynamic portfolio allocation
and asset trading decisions.
"""

import numpy as np
from collections import defaultdict


class PortfolioRLAgent:
    """
    RL agent for portfolio optimization using Q-learning.
    
    Parameters
    ----------
    n_assets : int
        Number of assets in portfolio
    learning_rate : float, default=0.1
        Learning rate
    discount_factor : float, default=0.95
        Discount factor for future rewards
    epsilon : float, default=0.1
        Exploration rate
    """
    
    def __init__(self, n_assets, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.n_assets = n_assets
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(n_assets + 1))  # +1 for hold action
        self.episode_rewards = []
        
    def _state_to_tuple(self, state):
        """Convert state to hashable tuple."""
        # Discretize state for Q-table
        discretized = np.round(state * 10) / 10
        return tuple(discretized)
        
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : array-like
            Current market state
        training : bool
            Whether in training mode (use exploration)
            
        Returns
        -------
        action : int
            Action index (which asset to trade or hold)
        """
        state_key = self._state_to_tuple(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_assets + 1)
        else:
            # Exploit: best action from Q-table
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
            
    def learn(self, state, action, reward, next_state, done=False):
        """
        Update Q-values based on experience.
        
        Parameters
        ----------
        state : array-like
            Previous state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : array-like
            New state
        done : bool
            Whether episode is finished
        """
        state_key = self._state_to_tuple(state)
        next_state_key = self._state_to_tuple(next_state)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
    def train(self, env, episodes=100):
        """
        Train the agent in an environment.
        
        Parameters
        ----------
        env : object
            Trading environment with step() and reset() methods
        episodes : int
            Number of training episodes
        """
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                
                self.learn(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
            self.episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")
                
    def policy(self, state):
        """
        Get optimal action without exploration.
        
        Parameters
        ----------
        state : array-like
            Current state
            
        Returns
        -------
        action : int
            Optimal action
        """
        return self.choose_action(state, training=False)
