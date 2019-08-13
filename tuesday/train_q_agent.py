#!/usr/bin/env python3
#
#  train_q_agent.py
#  Training Q-learning agent in OpenAI Gym's FrozenLake env
#
import random

import gym
from q_agent import QAgent
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# How long do we play
NUM_EPISODES = 500
# How often we print results
PRINT_EVERY_EPS = 100

environment = FrozenLakeEnv(is_slippery=False)

num_states = environment.observation_space.n
num_actions = environment.action_space.n

agent = QAgent(num_states, num_actions)

sum_reward = 0

for episode in range(NUM_EPISODES):
    done = False
    last_state = environment.reset()
    last_reward = None
    # Number of steps taken. A bit of a safeguard...
    num_steps = 0
    while not done:
        # Epsilon-greedy policy
        action = agent.get_action(last_state, environment)

        state, reward, done, info = environment.step(action)
        
        # A crude timeout: If we play too long without
        # completing the level, kill the game
        num_steps += 1
        if num_steps > 1000:
            print("Episode timeout! Could not finish in 1000 steps. Check your actions!")
            done = True

        # Update Q-table if we have one whole experience of
        # s, a, r, s', t'
        if last_state is not None:
            agent.update(
                last_state,
                action,
                reward,
                done,
                state,
            )

        last_state = state
        sum_reward += reward

        

    
    if (episode % PRINT_EVERY_EPS) == 0:
        print("Episode %d: %f" % (episode, sum_reward/PRINT_EVERY_EPS))
        sum_reward = 0

# Save the agent 
agent.save("q_table")