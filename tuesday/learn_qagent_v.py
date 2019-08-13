#!/usr/bin/env python3
#
#  learn_qagent_v.py
#  Learn value function of a trained Q-learning agent in FrozenLakeEnv
#
import gym
from v_table import VTable
from q_agent import QAgent
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# How long do we play
NUM_EPISODES = 100000
# How often we show current V-estimate
SHOW_EVERY_EPISODES = 10000

environment = FrozenLakeEnv(is_slippery=False)

num_states = environment.observation_space.n
num_actions = environment.action_space.n

vtable = VTable(num_states, discount_factor=0.5)
agent = QAgent(num_states, num_actions)
# Load already trained Q-table
agent.load("q_table.npy")

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
        # Take action according to Q-agent
        action = agent.get_action(state, environment)
        state, reward, done, info = environment.step(action)
        # Store reward
        rewards.append(reward)

    # Update v-estimate with the played game
    vtable.process_trajectory(states, rewards)

    if ((episode + 1) % SHOW_EVERY_EPISODES) == 0:
        vtable.visualize_v((4, 4))
