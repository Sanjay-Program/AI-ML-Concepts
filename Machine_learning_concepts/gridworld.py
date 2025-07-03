import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class GridWorld:
    def __init__(self, size=5, goal=(4, 4), walls=[]):
        self.size = size
        self.goal = goal
        self.walls = walls
        self.state_space = [(i, j) for i in range(size) for j in range(size) if (i, j) not in walls]
        self.action_space = ['up', 'down', 'left', 'right']

    def step(self, state, action):
        x, y = state
        if action == 'up':
            x = max(x - 1, 0)
        elif action == 'down':
            x = min(x + 1, self.size - 1)
        elif action == 'left':
            y = max(y - 1, 0)
        elif action == 'right':
            y = min(y + 1, self.size - 1)

        next_state = (x, y)
        if next_state in self.walls:
            next_state = state  # hit wall

        reward = 1 if next_state == self.goal else -0.1
        done = next_state == self.goal
        return next_state, reward, done

    def reset(self):
        return (0, 0)

# Encode states for use in Q-table
def encode_states(states):
    le = LabelEncoder()
    le.fit(states)
    return le, le.transform(states)

# Q-learning parameters
env = GridWorld(size=5, goal=(4, 4), walls=[(1, 1), (2, 2)])
le, encoded_states = encode_states(env.state_space)
n_states = len(encoded_states)
n_actions = len(env.action_space)
Q = np.zeros((n_states, n_actions))

episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

for ep in range(episodes):
    state = env.reset()
    done = False

    while not done:
        s_idx = le.transform([state])[0]

        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(n_actions)
        else:
            a_idx = np.argmax(Q[s_idx])

        action = env.action_space[a_idx]
        next_state, reward, done = env.step(state, action)
        ns_idx = le.transform([next_state])[0]

        # Q-learning update
        Q[s_idx, a_idx] = Q[s_idx, a_idx] + alpha * (reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, a_idx])

        state = next_state

# Print learned policy
policy_grid = [['' for _ in range(env.size)] for _ in range(env.size)]
for state in env.state_space:
    s_idx = le.transform([state])[0]
    best_action = env.action_space[np.argmax(Q[s_idx])]
    x, y = state
    policy_grid[x][y] = best_action[0].upper()

# Show policy grid
print("\nLearned Policy Grid:")
for row in policy_grid:
    print(' | '.join(a if a else '#' for a in row))
