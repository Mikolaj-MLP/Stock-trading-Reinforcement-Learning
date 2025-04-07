import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.vstack(states), actions, rewards, np.vstack(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# DRLAgent Class (DQN)
class DRLAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=5000, 
                 buffer_capacity=10000, batch_size=64, device='cpu'):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy exploration.
        state: a numpy vector of shape (state_dim,)
        """
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(q_values.argmax().item())
    
    def get_action(self, env):
        """
        Retrieves the current observation (which includes balance, shares, price, etc.)
        and returns the agent's selected action.
        (Note:if an action is invalid (e.g., sell when no shares are held), the environment will return a penalty.)
        """
        # Use the environment's observation function.
        state = env._get_observation()  # Expecting a vector: [balance, num_shares, current_price, total_asset_value, normalized_step]
        action = self.select_action(state)
        return int(action)
    
    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training Loop 

def train_dqn(agent, env, num_episodes=100, target_update=10, render_training=True, position_sizing_fn=None):
    """
    Train the DQN agent on the provided environment and visualize training progress.
    
    Parameters:
      agent: The DRLAgent instance.
      env: The trading environment.
      num_episodes: Number of episodes for training.
      target_update: Frequency (in episodes) to update the target network.
      render_training: If True, update a live plot of episode rewards.
      position_sizing_fn: Optional function to compute trade quantity.
                          It should accept (state, env, action, base_fraction) and return a quantity.
    
    Returns:
      agent: The trained agent.
      episode_rewards: List of total rewards per episode.
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get the action from the agent 
            action = agent.get_action(env)
            
            # Compute quantity using the provided position sizing function if available.
            if position_sizing_fn is not None:
                quantity = position_sizing_fn(state, env, action, base_fraction=0.05)
            else:
                quantity = None  # Default behavior from environment.
            
            next_state, reward, done, info = env.step((action, quantity))
            agent.push_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        # Update the target network periodically.
        if episode % target_update == 0:
            agent.update_target()
        
        if render_training:
            clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards, marker='o', linestyle='-')
            plt.xlabel("Episode")
            plt.ylabel("Episode Reward")
            plt.title("Training Progress (Episode Rewards)")
            plt.grid(True)
            plt.show()
            time.sleep(0.1)
        
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    return agent, episode_rewards
