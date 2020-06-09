# import gym
#
# env = gym.make('FrozenLake-v0')  # use the FrozenLake environment
#
# print(env.observation_space.n)   # get number of states
# print(env.action_space.n)   # get number of actions

# # basic syntax
# env.reset()  # reset environment to default state
# action = env.action_space.sample()  # get a random action
# new_state, reward, done, info = env.step(action)  # take action, notice it returns information about the action
# env.render()   # render the GUI for the environment


import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))  # create a matrix with all 0 values

EPISODES = 2000  # how many times to run the environment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of environment

LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96  # discount factor

RENDER = False

epsilon = 0.9  # start with a 90% chance of picking a random action

rewards = []
for episode in range(EPISODES):  # run however many times we have set

    state = env.reset()  # start by resetting
    for _ in range(MAX_STEPS):

        if RENDER:
            env.render()

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + LEARNING_RATE * (
                    reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards) / len(rewards)}:")
# and now we can see our Q values!

# we can plot the training progress and see how the agent improved
import matplotlib.pyplot as plt

def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
  avg_rewards.append(get_average(rewards[i:i+100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()

