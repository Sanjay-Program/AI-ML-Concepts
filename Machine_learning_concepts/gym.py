import gym
import numpy as np
import random
import time

# Initialize FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)  # Set is_slippery=True for harder version

# Q-table: rows = states, columns = actions
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.8          # Learning rate
gamma = 0.95         # Discount factor
epsilon = 1.0        # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000
max_steps = 100

# Training
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    for step in range(max_steps):
        # Choose action: explore or exploit
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action and observe
        next_state, reward, done, truncated, info = env.step(action)

        # Q-learning formula
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        if done:
            break

    # Decrease epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Show the final Q-table
print("\nTrained Q-table:")
print(np.round(q_table, 2))

# Test the agent
test_episodes = 5
print("\nTesting the trained agent...\n")

for episode in range(test_episodes):
    state = env.reset()[0]
    done = False
    print(f"Episode {episode + 1}\n")
    time.sleep(1)

    for step in range(max_steps):
        env.render()
        time.sleep(0.5)

        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)

        if done:
            env.render()
            if reward == 1:
                print("✅ Reached the Goal!")
            else:
                print("💀 Fell into a hole.")
            time.sleep(2)
            break

env.close()
