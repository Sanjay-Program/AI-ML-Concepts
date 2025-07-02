import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Step 1: Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Step 2: Hyperparameters and setup
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 500
batch_size = 64
memory = deque(maxlen=10000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = QNetwork(state_size, action_size).to(device)
target_net = QNetwork(state_size, action_size).to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

# Step 3: Function to choose action
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return random.randrange(action_size)
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.argmax(q_net(state)).item()

# Step 4: Experience Replay and Training
def train():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = q_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    target = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Step 5: Training loop
scores = []
for episode in range(episodes):
    state = env.reset()[0]
    score = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)
        memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        score += reward
        train()

    scores.append(score)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network every 10 episodes
    if episode % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Episode {episode}: Score = {score:.2f}, Epsilon = {epsilon:.3f}")

# Step 6: Plot the score
plt.plot(scores)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("DQN on CartPole")
plt.grid()
plt.show()
