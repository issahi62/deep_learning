#!/usr/bin/env python3
#
#  learn_random_v.py
#  Learning value function of random agent in FrozenLakeEnv
#
import gym
from v_table import VTable
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# How long do we play
NUM_EPISODES = 10000
# How often we show current V-estimate
SHOW_EVERY_EPISODES = 100

environment = FrozenLakeEnv(is_slippery=False)

num_states = environment.observation_space.n

# Create a tabular record of values
vtable = VTable(num_states)

for episode in range(NUM_EPISODES):
    done = False
    state = environment.reset()
    # Keep track of visited states and rewards
    # obtained
    states = []
    rewards = []
    while not done:
        # Store state
        states.append(state)
        # Take random action
        state, reward, done, info = environment.step(
            environment.action_space.sample()
        )
        # Store reward
        rewards.append(reward)

    # Update v-estimate with the played game
    vtable.process_trajectory(states, rewards)

    if ((episode + 1) % SHOW_EVERY_EPISODES) == 0:
        vtable.visualize_v((4, 4))
