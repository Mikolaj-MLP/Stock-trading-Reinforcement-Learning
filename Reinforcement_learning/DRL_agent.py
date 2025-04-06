import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Simple Replay Buffer.
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

# basic DQN agent.
class DRLAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=5000, 
                 buffer_capacity=10000, batch_size=64, device='cpu'):
        self.device = torch.device(device)
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
        """Select an action using epsilon-greedy exploration."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
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
        Get an action from the DRL agent given the current environment state.
        This method retrieves the state from the environment and applies safety checks:
          - If the agent selects 'buy' but doesn't have enough balance to buy one share,
            the action is overridden to 'hold'.
          - If the agent selects 'sell' but holds no shares, the action is overridden to 'hold'.
        """
        state = env._get_observation()  
        action = self.select_action(state)
        
        # Safety checks:
        balance = state[0]
        num_shares = state[1]
        current_price = state[2]
        
        # If action is Buy (1) but not enough balance to buy one share.
        if action == 1 and balance < current_price:
            action = 0  # Hold
        # If action is Sell (2) but no shares held.
        if action == 2 and num_shares <= 0:
            action = 0  # Hold
        
        return int(action)
    
    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """Sample a batch and update the policy network."""
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
        """Update the target network with the policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# A simple training loop for the agent
def train_dqn(agent, env, num_episodes=100, target_update=10):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(env)
            next_state, reward, done, info = env.step((action, None))
            agent.push_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward
        
        if episode % target_update == 0:
            agent.update_target()
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    return agent
