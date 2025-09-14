import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import SimpleDQN, DuelingDQN

class DQNAgent:
    """Interacts with and learns from the environment."""
    def __init__(self, n_states: int, n_actions: int, config):
        self.n_actions = n_actions
        self.config = config
        self.device = config.DEVICE

        if config.dueling_network:
            self.policy_net = DuelingDQN(n_states, config.HIDDEN_LAYER_SIZE, n_actions).to(self.device)
            self.target_net = DuelingDQN(n_states, config.HIDDEN_LAYER_SIZE, n_actions).to(self.device)
        else:
            self.policy_net = SimpleDQN(n_states, config.HIDDEN_LAYER_SIZE, n_actions).to(self.device)
            self.target_net = SimpleDQN(n_states, config.HIDDEN_LAYER_SIZE, n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.loss_func = nn.MSELoss()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(self.n_actions)

    def optimize_model(self, memory):
        if len(memory) < self.config.BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = memory.sample(self.config.BATCH_SIZE)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(-1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(-1)

        expected_q_values = rewards + (self.config.GAMMA * next_q_values * (1 - dones))
        loss = self.loss_func(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
