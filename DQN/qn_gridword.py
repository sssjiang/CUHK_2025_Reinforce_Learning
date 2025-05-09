import numpy as np
import random

class GridWorld:
    def __init__(self, grid_size=4, start_pos=(0, 0), goal_pos=(3, 3), trap_pos=[(1, 1), (3, 2)]):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.trap_pos = trap_pos
        self.agent_pos = start_pos
        self.actions = ["up", "down", "left", "right"]
        self.num_actions = len(self.actions)
        self.num_states = grid_size * grid_size
        self.state_map = {(r, c): i for i, (r, c) in enumerate([(r, c) for r in range(grid_size) for c in range(grid_size)])}
        self.reverse_state_map = {i: (r, c) for (r, c), i in self.state_map.items()}

    def get_state(self):
        return self.state_map[self.agent_pos]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.get_state()

    def step(self, action_index):
        action = self.actions[action_index]
        row, col = self.agent_pos
        new_row, new_col = row, col

        if action == "up":
            new_row = max(0, row - 1)
        elif action == "down":
            new_row = min(self.grid_size - 1, row + 1)
        elif action == "left":
            new_col = max(0, col - 1)
        elif action == "right":
            new_col = min(self.grid_size - 1, col + 1)

        self.agent_pos = (new_row, new_col)
        reward = -1  # Default reward for each step
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        elif self.agent_pos in self.trap_pos:
            reward = -10
            done = True

        return self.get_state(), reward, done

def q_learning(env, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_rate=0.001, num_episodes=5000):
    q_table = np.zeros((env.num_states, env.num_actions))
    epsilon = epsilon_start
    rng = np.random.default_rng()

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action_index = rng.integers(env.num_actions)  # Explore
            else:
                action_index = np.argmax(q_table[state, :])  # Exploit

            next_state, reward, done = env.step(action_index)

            # Q-value update rule
            best_next_q = np.max(q_table[next_state, :])
            q_table[state, action_index] = q_table[state, action_index] + alpha * (reward + gamma * best_next_q - q_table[state, action_index])

            state = next_state

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)

    return q_table

def play_episode(env, q_table, max_steps=20):
    state = env.reset()
    path = [env.reverse_state_map[state]]
    total_reward = 0
    for _ in range(max_steps):
        action_index = np.argmax(q_table[state, :])
        next_state, reward, done = env.step(action_index)
        path.append(env.reverse_state_map[next_state])
        total_reward += reward
        state = next_state
        if done:
            break
    print(f"Path: {path}, Total Reward: {total_reward}")

if __name__ == "__main__":
    env = GridWorld()
    learned_q_table = q_learning(env)

    print("Learned Q-table:")
    print(learned_q_table)

    print("\nPlaying a few episodes based on the learned Q-table:")
    for _ in range(5):
        play_episode(env, learned_q_table)