def risk_management(state, env, action, quantity):
    """
    to be replaced by risk estimating function
    """
    return action, quantity

def max_loss_risk_management(state, env, action, quantity, loss_threshold=0.90):
    """
    Forces a sell if total asset value drops below a fraction (loss_threshold)
    of the maximum asset value seen so far.
    If the current total asset value is below loss_threshold * max_asset_value, 
    it overrides any signal to force a full sell.
    """
    # Initialize or update the maximum asset value.
    if not hasattr(env, 'max_asset_value'):
        env.max_asset_value = env.total_asset_value
    else:
        env.max_asset_value = max(env.total_asset_value, env.max_asset_value)
    
    if env.total_asset_value < loss_threshold * env.max_asset_value:
        return 2, env.num_shares  # Force a full sell.
    return action, quantity

def position_limit_risk_management(state, env, action, quantity, max_position_fraction=0.5):
    """
    Limits the position size relative to the total asset value.
    If the current position (in dollars) is above max_position_fraction of the total asset value,
    then override a buy signal with a hold.
    """
    current_price = state[2]
    position_value = env.num_shares * current_price
    if action == 1 and position_value >= max_position_fraction * env.total_asset_value:
        return 0, 0  # Override buy with hold.
    return action, quantity
