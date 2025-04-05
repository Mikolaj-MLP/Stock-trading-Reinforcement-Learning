import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    A flexible Gym environment for stock trading that accepts a trade tuple (action, quantity).
    Transaction cost is applied on each trade.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_cost_pct=0.001, custom_render=None):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.custom_render = custom_render
        
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        
        # Action: 0=Hold, 1=Buy, 2=Sell.
        self.action_space = spaces.Discrete(3)
        # Observation: [balance, num_shares, current_price, total_asset_value, normalized_step]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.trade_history = []  # Records of trades.
        self.reset()

    def _get_current_price(self):
        return float(self.df.loc[self.current_step, 'Close'])

    def _get_observation(self):
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

    def _take_action(self, action, quantity):
        current_price = self._get_current_price()
        
        if action == 1:  # Buy
            if quantity is None:
                quantity = int(self.balance // (current_price * (1+self.transaction_cost_pct)))
            if quantity <= 0:
                return
            cost = quantity * current_price * (1 + self.transaction_cost_pct)
            if self.balance >= cost:
                self.balance -= cost
                self.num_shares += quantity
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': current_price,
                    'quantity': quantity
                })
        elif action == 2:  # Sell
            if quantity is None:
                quantity = self.num_shares
            if quantity > self.num_shares:
                quantity = self.num_shares
            if quantity <= 0:
                return
            revenue = quantity * current_price * (1 - self.transaction_cost_pct)
            self.balance += revenue
            self.num_shares -= quantity
            self.trade_history.append({
                'step': self.current_step,
                'action': 'sell',
                'price': current_price,
                'quantity': quantity
            })
        # Hold (action == 0): do nothing.
        
        self.total_asset_value = self.balance + self.num_shares * current_price

    def step(self, action_tuple):
        """
        Expects a tuple (action, quantity). Returns (observation, reward, done, info).
        """
        action, quantity = action_tuple
        prev_total = self.total_asset_value
        self._take_action(action, quantity)
        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = self._get_observation()
        reward = self.total_asset_value - prev_total
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
        self.trade_history = []
        return self._get_observation()

    def render(self, mode='human'):
        if self.custom_render:
            self.custom_render(self)
        else:
            print(f"Step: {self.current_step}")
            print(f"Balance: {self.balance:.2f} | Shares: {self.num_shares} | Total Asset: {self.total_asset_value:.2f}")
            print(f"Current Price: {self._get_current_price():.2f}\n")
