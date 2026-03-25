import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay_rate=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state, :])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = new_value

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

    def train(self, num_episodes):
        rewards_per_episode = []
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
            rewards_per_episode.append(total_reward)
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {self.epsilon:.2f}")
        return rewards_per_episode

if __name__ == "__main__":
    import gymnasium as gym

    # Create a simple environment (e.g., FrozenLake)
    env = gym.make("FrozenLake-v1", is_slippery=False)

    agent = QLearningAgent(env, alpha=0.8, gamma=0.95, epsilon=1.0, epsilon_decay_rate=0.999, min_epsilon=0.05)
    rewards = agent.train(num_episodes=2000)

    print("\nTraining complete. Average reward over last 100 episodes:", np.mean(rewards[-100:]))

    # Test the trained agent
    state = env.reset()[0]
    done = False
    print("\nTesting trained agent:")
    for step in range(10):
        action = np.argmax(agent.q_table[state, :])
        next_state, reward, done, _, _ = env.step(action)
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
        state = next_state
        if done:
            break
    env.close()
