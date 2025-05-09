import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)
class GridWorldEnv:
    def __init__(self, size=5, goal=(4, 4), obstacle=(2, 2)):
        self.size = size
        self.goal = goal
        self.obstacle = obstacle
        self.agent_pos = (0, 0)
        self.action_space = [0, 1, 2, 3]  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = (size, size)

    def reset(self):
        self.agent_pos = (0, 0)
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.agent_pos] = 1.0  # Agent position
        state[self.goal] = 0.5      # Goal position (different value for visualization)
        state[self.obstacle] = -0.5  # Obstacle position (different value for visualization)
        return state.flatten()      # Flatten the state for the network

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # Up
            new_pos = (max(0, x - 1), y)
        elif action == 1:  # Down
            new_pos = (min(self.size - 1, x + 1), y)
        elif action == 2:  # Left
            new_pos = (x, max(0, y - 1))
        elif action == 3:  # Right
            new_pos = (x, min(self.size - 1, y + 1))
        else:
            raise ValueError("Invalid action")

        self.agent_pos = new_pos
        reward = 0
        done = False
        if self.agent_pos == self.goal:
            reward = 1
            done = True
        elif self.agent_pos == self.obstacle:
            reward = -1
            done = True

        return self._get_state(), reward, done, {}

    def render(self):
        grid = np.zeros((self.size, self.size))
        grid[self.agent_pos] = 1
        grid[self.goal] = 0.5
        grid[self.obstacle] = -0.5
        plt.imshow(grid, cmap='viridis')
        plt.title(f"Agent at {self.agent_pos}")
        plt.pause(0.1)
        plt.clf()

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        if random.random() > self.epsilon:
            return torch.argmax(action_values).item()  # Exploit
        else:
            return random.choice(range(self.action_size))  # Explore

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        # print(states)
        # print(actions)
        # print(rewards)
        # print(next_states)
        # print(dones)
        # breakpoint()

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(list(actions), dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(list(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(list(dones), dtype=torch.uint8).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0) # Gradient clipping
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    env = GridWorldEnv(size=5)
    state_size = env.observation_space[0] * env.observation_space[1]
    print(state_size)
    action_size = len(env.action_space)
    print(action_size)
    agent = DQNAgent(state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999)

    num_episodes = 500
    target_update_interval = 10
    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            step += 1
            # env.render() # Uncomment to visualize the agent

        rewards_history.append(total_reward)
        if episode % target_update_interval == 0:
            agent.update_target_network()
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Plotting rewards
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Rewards")
    plt.savefig("rewards.png") 
     # 策略评估循环 (输出最优解)
    print("\nEvaluating learned policy:")
    env.reset()  # 重置环境到初始状态
    state = env._get_state()
    done = False
    path = [env.agent_pos]  # 记录智能体的路径

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        agent.q_network.eval()  # 设置为评估模式
        with torch.no_grad():
            action_values = agent.q_network(state_tensor)
        action = torch.argmax(action_values).item()  # 选择具有最高 Q 值的动作
        agent.q_network.train() # 恢复训练模式

        next_state, reward, done, _ = env.step(action)
        path.append(env.agent_pos)
        state = next_state
        env.render() # 可视化智能体的行动

        if done:
            print(f"Goal reached! Path: {path}")
            break
        elif env.agent_pos == env.obstacle:
            print(f"Hit obstacle! Path: {path}")
            break
        elif len(path) > env.size * env.size * 2: # 防止智能体卡在循环中
            print(f"Stuck! Path: {path}")
            break
    # plt.show()