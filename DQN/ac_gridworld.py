import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Environment (same as in dqn_gridworld.py)
class GridWorldEnv:
    def __init__(self, size=5, goal=(4, 4), obstacle=(2, 2)):
        self.size = size
        self.goal = goal
        self.obstacle = obstacle
        self.agent_pos = (0, 0)
        self.action_space_n = 4  # 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = [0, 1, 2, 3] 
        self.observation_space_shape = (size * size,)


    def reset(self):
        self.agent_pos = (0, 0)
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.agent_pos] = 1.0
        state[self.goal] = 0.5
        state[self.obstacle] = -0.5
        return state.flatten()

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
        
        # Small negative reward for each step to encourage efficiency
        if not done:
            reward = -0.01

        return self._get_state(), reward, done, {}

    def render(self, filename=None):
        grid = np.zeros((self.size, self.size))
        grid[self.agent_pos] = 1.0
        grid[self.goal] = 0.5
        grid[self.obstacle] = -0.5
        
        fig, ax = plt.subplots() 
        ax.imshow(grid, cmap='viridis')
        ax.set_title(f"Agent at {self.agent_pos}")
        
        if filename:
            plt.savefig(filename)
        else: # If no filename, try to show (might not work in SSH without X11)
            plt.show(block=False)
            plt.pause(0.1)
        
        plt.close(fig)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size) # Outputs logits for actions

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1) # Outputs state value V(s)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_size, action_size, actor_lr=0.001, critic_lr=0.001, gamma=0.99, entropy_coeff=0.01, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.buffer = deque(maxlen=buffer_size)

    def remember(self, state, action, reward, next_state, done):
        # print(f"Remembering: {state}, {action}, {reward}, {next_state}, {done}")
        # # stop
        # exit()
        self.buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.actor.eval() # Set actor to evaluation mode
        with torch.no_grad():
            action_logits = self.actor(state)
            dist = Categorical(logits=action_logits)
            action = dist.sample().item()
        self.actor.train() # Set actor back to training mode
        return action

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(list(actions), dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(list(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(list(dones), dtype=torch.uint8).unsqueeze(1).to(self.device)

        # --- Critic Update ---
        current_values = self.critic(states)  # V(s)
        next_values = self.critic(next_states) # V(s')
        
        # TD Target: R + gamma * V(s') * (1 - done)
        # We detach next_values as it's part of the target, not to be trained here
        td_target = rewards + self.gamma * next_values.detach() * (1 - dones)
        
        # Critic loss: MSE between V(s) and TD Target
        critic_loss = F.mse_loss(current_values, td_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Advantage: TD Target - V(s)
        # We detach current_values from advantage calculation for actor, or use td_target - current_values.detach()
        # The advantage should not backpropagate gradients into the critic during actor's update
        advantage = (td_target - current_values).detach() 
        
        action_logits = self.actor(states)
        dist = Categorical(logits=action_logits)
        log_probs_taken_action = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        
        entropy = dist.entropy().mean() # Entropy of the policy distribution
        
        # Actor loss: -log_prob(a|s) * Advantage - entropy_bonus
        actor_loss = (-log_probs_taken_action * advantage - self.entropy_coeff * entropy).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

if __name__ == "__main__":
    env = GridWorldEnv(size=5)
    state_size = env.observation_space_shape[0]
    action_size = env.action_space_n
    
    agent = ActorCriticAgent(state_size, action_size, entropy_coeff=0.01)

    num_episodes = 1000 # Actor-Critic might need more episodes
    rewards_history = []
    avg_rewards_history = []

    print(f"Training on device: {agent.device}")

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn() # Learn at each step if buffer is full enough
            
            state = next_state
            total_reward += reward
            # env.render() # Uncomment to visualize training, might be slow

        rewards_history.append(total_reward)
        
        # Calculate moving average for better trend visualization
        if episode >= 99:
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards_history.append(avg_reward)
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")
        else:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    # Plotting rewards
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards")

    plt.subplot(1, 2, 2)
    if avg_rewards_history:
        plt.plot(range(99, num_episodes), avg_rewards_history)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward (Last 100 Episodes)")
        plt.title("Moving Average of Rewards")
    
    plt.tight_layout()
    plt.savefig("ac_rewards.png")
    print("\nTraining finished. Rewards plot saved to ac_rewards.png")

    # Evaluate learned policy
    print("\nEvaluating learned policy:")
    env.reset()
    state = env._get_state()
    done = False
    eval_path = [env.agent_pos]
    eval_total_reward = 0
    eval_steps = 0
    
    # Create a directory for evaluation frames if it doesn't exist
    import os
    eval_frames_dir = "ac_eval_frames"
    if not os.path.exists(eval_frames_dir):
        os.makedirs(eval_frames_dir)

    while not done and eval_steps < env.size * env.size * 2 : # Max steps to prevent infinite loops
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        agent.actor.eval()
        with torch.no_grad():
            action_logits = agent.actor(state_tensor)
            # For evaluation, typically take the most likely action (argmax)
            action = torch.argmax(action_logits, dim=1).item()
            # Alternatively, can still sample:
            # dist = Categorical(logits=action_logits)
            # action = dist.sample().item()
        agent.actor.train()

        next_state, reward, done, _ = env.step(action)
        eval_path.append(env.agent_pos)
        state = next_state
        eval_total_reward += reward
        eval_steps += 1
        
        env.render(filename=os.path.join(eval_frames_dir, f"eval_frame_{eval_steps:03d}.png"))

    if done and env.agent_pos == env.goal:
        print(f"Goal reached in {eval_steps} steps! Total reward: {eval_total_reward:.2f}. Path: {eval_path}")
    elif env.agent_pos == env.obstacle:
        print(f"Hit obstacle after {eval_steps} steps. Total reward: {eval_total_reward:.2f}. Path: {eval_path}")
    else:
        print(f"Evaluation finished after {eval_steps} (max) steps. Total reward: {eval_total_reward:.2f}. Path: {eval_path}") 