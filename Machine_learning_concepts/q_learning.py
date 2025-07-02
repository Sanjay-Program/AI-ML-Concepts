import numpy as np
import gym

# Step 1: Load the FrozenLake environment (4x4 grid)
env = gym.make("FrozenLake-v1", is_slippery=False)  # deterministic for easier learning

# Step 2: Initialize Q-table
state_size = env.observation_space.n  # number of states
action_size = env.action_space.n      # number of actions
Q = np.zeros((state_size, action_size))

# Step 3: Set hyperparameters
alpha = 0.8        # learning rate
gamma = 0.95       # discount factor
epsilon = 0.1      # exploration rate
episodes = 1000    # number of training episodes

# Step 4: Q-learning algorithm
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        # ε-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(Q[state, :])     # exploit

        # Take the action
        next_state, reward, done, truncated, info = env.step(action)

        # Q-learning update rule
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state

# Step 5: Display learned Q-table
np.set_printoptions(precision=2)
print("Learned Q-table:")
print(Q)

# Step 6: Evaluate the learned policy
state = env.reset()[0]
env.render()
done = False
print("\nPolicy execution:")

while not done:
    action = np.argmax(Q[state])
    state, reward, done, truncated, _ = env.step(action)
    env.render()
