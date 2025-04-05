import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    A flexible Gym environment for stock trading with trade history recording.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_cost_pct=0.001,
                 reward_function=None, custom_render=None):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_function = reward_function
        self.custom_render = custom_render
        
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [balance, num_shares, current_price, total_asset_value, normalized_step]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        
        # To record trade actions for visualization
        self.trade_history = []
        
        self.reset()

    def _get_current_price(self):
        """Retrieve the current stock price as a float."""
        return float(self.df.loc[self.current_step, 'Close'])

    def _get_observation(self):
        """Construct the observation vector with scalar floats."""
        current_price = self._get_current_price()
        normalized_step = float(self.current_step) / float(self.max_steps) if self.max_steps > 0 else 0.0
        obs = np.array([
            float(self.balance),
            float(self.num_shares),
            current_price,
            float(self.total_asset_value),
            normalized_step
        ], dtype=np.float32)
        return obs

    def _take_action(self, action):
        """
        Execute the given action and record trades.
        Action: 0 = Hold, 1 = Buy, 2 = Sell
        """
        current_price = self._get_current_price()
        
        if action == 1:  
            max_shares = int(self.balance // current_price)
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_cost_pct)
                if self.balance >= cost:
                    self.balance -= cost
                    self.num_shares += max_shares
                    # Record the transaction
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': 1,
                        'price': current_price,
                        'shares': max_shares
                    })
                    
        elif action == 2:  
            if self.num_shares > 0:
                revenue = self.num_shares * current_price * (1 - self.transaction_cost_pct)
                self.balance += revenue
                # Record the transaction
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 2,
                    'price': current_price,
                    'shares': self.num_shares
                })
                self.num_shares = 0

        self.total_asset_value = self.balance + self.num_shares * current_price

    def step(self, action):
        previous_total = self.total_asset_value
        
        self._take_action(action)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        obs = self._get_observation()
        
        if self.reward_function:
            reward = self.reward_function(self, action, previous_total, self.total_asset_value)
        else:
            reward = self.total_asset_value - previous_total
        
        info = {
            'balance': self.balance,
            'num_shares': self.num_shares,
            'total_asset_value': self.total_asset_value,
            'current_price': self._get_current_price()
        }
        
        return obs, reward, done, info

    def reset(self):
        self.balance = self.initial_balance
        self.num_shares = 0
        self.total_asset_value = self.initial_balance
        self.current_step = 0
        self.trade_history = []  # Reset trade history
        return self._get_observation()

    def render(self, mode='human'):
        if self.custom_render:
            self.custom_render(self)
        else:
            # Default render prints out the state
            print(f"Step: {self.current_step}")
            print(f"Balance: {self.balance:.2f} | Shares: {self.num_shares} | Total Asset: {self.total_asset_value:.2f}")
            print(f"Current Price: {self._get_current_price():.2f}\n")
