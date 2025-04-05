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