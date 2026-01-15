import numpy as np
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    """
    A simplified Portfolio Optimization environment for Banking/Finance.
    The agent manages a portfolio of assets to maximize returns while minimizing risk.
    """
    def __init__(self, num_assets=5, window_size=10):
        super(PortfolioEnv, self).__init__()
        self.num_assets = num_assets
        self.window_size = window_size
        
        # Action: weights for each asset (must sum to 1, but we'll normalize in step)
        self.action_space = spaces.Box(low=0, high=1, shape=(num_assets,), dtype=np.float32)
        
        # State: recent returns of assets
        self.observation_space = spaces.Box(low=-1, high=1, shape=(num_assets * window_size,), dtype=np.float32)
        
        self.data = self._generate_synthetic_data()
        self.current_step = window_size

    def _generate_synthetic_data(self, length=2000):
        # Generate synthetic asset returns with some correlation and trends
        np.random.seed(42)
        # Base returns: some assets are better than others
        base_returns = np.array([0.001, 0.0008, 0.0005, 0.0002, -0.0001])
        returns = np.random.normal(base_returns, 0.01, (length, self.num_assets))
        # Add some time-varying trends
        t = np.linspace(0, 20, length)
        for i in range(self.num_assets):
            returns[:, i] += np.sin(t + i) * 0.002
        return returns

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        return self._get_obs(), {}

    def _get_obs(self):
        return self.data[self.current_step - self.window_size:self.current_step].flatten()

    def step(self, action):
        # Normalize action to sum to 1
        weights = action / (np.sum(action) + 1e-8)
        
        # Calculate portfolio return
        asset_returns = self.data[self.current_step]
        portfolio_return = np.dot(weights, asset_returns)
        
        # Reward: Return - Risk (variance penalty)
        # In a real scenario, we'd use a rolling variance
        risk_penalty = 0.1 * np.var(weights * asset_returns)
        reward = (portfolio_return - risk_penalty) * 1000  # Scale up for better learning
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_obs(), reward, done, False, {}

def collect_offline_data(env, num_episodes=100):
    """Collects 'medium' quality data using a noisy random policy."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    states, actions, next_states, rewards, dones = [], [], [], [], []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # Noisy policy: biased towards some assets but with noise
            action = np.random.dirichlet(np.ones(action_dim)) + np.random.normal(0, 0.1, action_dim)
            action = np.clip(action, 0, 1)
            
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            
    return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)
