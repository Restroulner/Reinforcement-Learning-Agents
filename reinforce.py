import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    A simple policy network for REINFORCE algorithm.
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    """
    def __init__(self, state_size, action_size, learning_rate=0.01, discount_factor=0.99):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor

        self.rewards = []
        self.log_probs = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        discounts = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.discount_factor * R
            discounts.insert(0, R)
        
        discounts = torch.tensor(discounts)
        discounts = (discounts - discounts.mean()) / (discounts.std() + 1e-9) # Normalize rewards

        policy_gradient = []
        for log_prob, G in zip(self.log_probs, discounts):
            policy_gradient.append(-log_prob * G)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_gradient).sum()
        loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []

if __name__ == "__main__":
    # Simple environment (e.g., a simplified CartPole-like scenario)
    # State: [position, velocity], Action: [left, right]
    state_size = 2
    action_size = 2

    agent = REINFORCEAgent(state_size, action_size, learning_rate=0.001)

    # Simulate training over episodes
    n_episodes = 500
    max_steps_per_episode = 100

    print("Starting REINFORCE training simulation...")
    for episode in range(n_episodes):
        state = np.random.rand(state_size) * 2 - 1 # Random initial state
        episode_rewards = 0
        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            
            # Simulate environment step (dummy for demonstration)
            next_state = state + np.random.rand(state_size) * 0.1 * (1 if action == 1 else -1)
            reward = 1 if np.sum(next_state) > 0 else -1 # Simple reward logic
            done = (step == max_steps_per_episode - 1) or (abs(np.sum(next_state)) > 2)

            agent.store_reward(reward)
            state = next_state
            episode_rewards += reward

            if done:
                break
        
        agent.update_policy()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - Total Reward: {episode_rewards}")

    print("REINFORCE training simulation finished.")
