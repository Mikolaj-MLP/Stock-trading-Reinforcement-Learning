class PercentChangeStrategy:
    """
    Strategy that uses a reference price.
    When not holding, it will trigger a buy if the current price is below a threshold relative to the reference.
    When holding, it triggers a sell only if the current price exceeds the reference by a threshold.
    """
    def __init__(self, threshold_down=0.95, threshold_up=1.10, initial_reference=None):
        self.threshold_down = threshold_down
        self.threshold_up = threshold_up
        self.reference_price = initial_reference

    def get_action(self, env):
        current_price = float(env.df.loc[env.current_step, 'Close'])
        # Initialize reference price if not set.
        if self.reference_price is None:
            self.reference_price = current_price
        
        # If holding a position, sell only if price is sufficiently above the reference.
        if env.num_shares > 0:
            if current_price >= self.reference_price * self.threshold_up:
                self.reference_price = current_price
                return 2  # Sell
            else:
                return 0  # Hold
        else:
            # If not holding, buy if the price is sufficiently below the reference.
            if current_price <= self.reference_price * self.threshold_down:
                self.reference_price = current_price
                return 1  # Buy
            else:
                return 0  # Hold


class ExtremaStrategy:
    """
    Strategy that uses a window of past prices.
    If the current price is the minimum in the window, it signals a Buy.
    If it is the maximum, it signals a Sell.
    Otherwise, it holds.
    """
    def __init__(self, window=10):
        self.window = window

    def get_action(self, env):
        if env.current_step < self.window:
            return 0  # Not enough data.
        window_prices = env.df['Close'].iloc[env.current_step - self.window: env.current_step + 1].values.astype(float)
        current_price = float(env.df.loc[env.current_step, 'Close'])
        if current_price <= window_prices.min():
            return 1  # Buy
        elif current_price >= window_prices.max():
            return 2  # Sell
        else:
            return 0  # Hold
