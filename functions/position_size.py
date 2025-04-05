def all_in_position_sizing(state, env, action):
    """
    All-In Position Sizing:
      - For Buy: use all available balance to buy shares.
      - For Sell: sell all held shares.
    Returns the quantity to trade.
    """
    current_price = state[2]
    if action == 1:  # Buy
        qty = int(env.balance // current_price)
        return qty
    elif action == 2:  # Sell
        return env.num_shares
    return 0

def fixed_fraction_position_sizing(state, env, action, fraction=0.10):
    """
    Uses a fixed fraction of available resources.
      - For Buy: invest a fraction of available balance.
      - For Sell: sell a fraction of held shares.
    """
    current_price = state[2]
    if action == 1:  # Buy
        budget = fraction * env.balance
        qty = int(budget // current_price)
        return qty
    elif action == 2:  # Sell
        qty = int(fraction * env.num_shares)
        return qty if qty > 0 else env.num_shares
    return 0

def scale_in_position_sizing(state, env, action, base_fraction=0.05):
    """
    Scaling-in strategy for consecutive buy decisions:
      - If the previous trades were buys, increase the fraction of available capital.
      - For Sell, use a fixed fraction (e.g. 50% of the holdings).
    """
    current_price = state[2]
    if action == 1:  # Buy
        # Count consecutive buys in trade history.
        consecutive_buys = 0
        for trade in reversed(env.trade_history):
            if trade['action'] == 'buy':
                consecutive_buys += 1
            else:
                break
        # Increase fraction with each consecutive buy.
        fraction = base_fraction * (1 + consecutive_buys)
        budget = fraction * env.balance
        qty = int(budget // current_price)
        return qty
    elif action == 2:  # Sell
        qty = int(0.5 * env.num_shares)
        return qty if qty > 0 else env.num_shares
    return 0
