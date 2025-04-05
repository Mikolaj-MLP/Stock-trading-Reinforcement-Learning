def state_transformation(state):
    """
    Feature extraction from the raw observation.
    For now, return the state unchanged.
    """
    return state

def logging_callback(step, state, action, quantity, reward, info):
    """
    Logging callback to output step-by-step details.
    """
    print(f"Step: {step}, Action: {action}, Quantity: {quantity}, Reward: {reward:.2f}")