#!/usr/bin/env python3
#
#  agent_sarsa.py
#  Implementation of tabular Q-learning
#
import numpy as np
import math as m
import random
from matplotlib import pyplot

# Hardcoded default values
DISCOUNT_FACTOR = 0.98
LEARNING_RATE = 0.1
# Initial values in Q-table
INIT_VALUE = random.random()

class QAgent:
    """
    Q-learning agent
    """
    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate=LEARNING_RATE,
                 discount_factor=DISCOUNT_FACTOR,
                 init_value=INIT_VALUE):
        # Table storing Q-values
        #self.q = np.ones((num_states, num_actions)) #* init_value
        self.q = np.random.random((num_states, num_actions))
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, state, env):
        """
        Returns action according to greedy policy:
        Action with the highest value.
        Parameters:
            state - Int indicating current state
            env - Gym environment we are playing
        Returns:
            action - Int indicating optimal action
        """
        # Greedy policy: Take action with
        # highest Q-value in that state.
        # argmax returns _index_ of the maximum value.

        return np.argmax(self.q[state])

    def update(self, s, a, r, t, s_prime):
        """
        Update Q-values with Q-learning
        Parameters:
            s,a,r - State, action and reward at time t
            t - If s_prime is terminal
            s_prime - State at time t+1
        """
        # Q-learning update:
         #Q(s,a) = Q(s,a) + \alpha * [r + \gamma * \max_a Q(s',a) - Q(s,a)]

        #raise NotImplementedError("Implement Q-learning update")

        # TODO Q-learning update
        # Two parts:
        target = 0
        if t == True:
            target = r
        else:
            #  1) Compute the target value reward + DISCOUNT * \max_a Q(s', a)
            target =  r + DISCOUNT_FACTOR * np.max(self.q[s_prime])
        #  2) Update Q-values with Q(s, a) += LEARNING_RATE * (target - Q(s, a))
        self.q[s,a] += LEARNING_RATE * (target - self.q[s, a])

        # 1) Compute target
        # Note: If s_prime (s') is a terminal state (t), then target is only "target = reward"
        #       (You will need an if-else struct)
       

        # 2) Update Q-values
       

    def visualize_q(self, grid_size, show_max_qs=False):
        """
        Visualize Q-values with matplotlib by arranging
        observations into a grid of given size.
        Shown values are either average over all Q-values
        (default), or maximum Q-values if show_max_qs=True
        """
        q_values = None
        if show_max_qs:
            # Maximum over actions
            q_values = self.q.max(axis=1)
        else:
            # Mean over actions
            q_values = self.q.mean(axis=1)
        # Reshape to match with the grid we have
        q_values = q_values.reshape(grid_size)
        pyplot.imshow(q_values)
        pyplot.show()

    def save(self, path):
        """Save learned Q-table to given path.

        Parameters:
            path - String, where to save current Q-table
        """
        np.save(path, self.q)

    def load(self, path):
        """Load Q-table from given path

        Parameters:
            path - String, where to load Q-table from
        """
        self.q = np.load(path)
