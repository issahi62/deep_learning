#!/usr/bin/env python3
#
#  agent_sarsa.py
#  Implementation of tabular Q-learning
#
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot

# Hardcoded default values
DISCOUNT_FACTOR = 0.98


class VTable:
    """
    Simple V-function learning based on tables
    """
    def __init__(self,
                 num_states,
                 discount_factor=DISCOUNT_FACTOR):
        # Table storing sum-of-returns for each
        # state
        self.return_sums = np.zeros((num_states, ))
        # How often we have seen the state.
        # Start from one visit to each state to avoid
        # div-by-zero
        self.num_visits = np.ones((num_states, ), dtype=np.long)
        self.discount_factor = discount_factor

    def process_trajectory(self, states, rewards):
        """
        Update value function estimates with given trajectory
        of rewards and states
        Parameters:
            states - List of length N of states visited
            rewards - List of length N of rewards in trajectory
        """

        # Start going over states from end to beginning,
        # updating the return as we go.
        current_return = 0

        # From final step to first
        for i in range(len(states) - 1, -1, -1):
            current_return = rewards[i] + self.discount_factor * current_return
            # Update arrays used to compute value function
            state = states[i]
            self.return_sums[state] += current_return
            self.num_visits[state] += 1

    def get_v(self):
        """
        Return current estimate of the value function
        """
        #raise NotImplementedError("Implement get_v in v_table.py")

        # TODO 
        # Return estimate of value of states.
        # This is average of _all_ possible returns, starting 
        # from this state.
        # Note that there are two variables available:
        #   - self.return_sums: Contains sum of all returns seen so far
        #   - self.num_visits: How often that state has been visited so far
        estimate = self.return_sums / self.num_visits
        value =  estimate #"this is bogus. Replace with proper value"
        return value

    def visualize_v(self, grid_size):
        """
        Visualize V-values with matplotlib by arranging
        observations into a grid of given size.
        """
        v_values = self.get_v()
        v_values = v_values.reshape(grid_size)
        pyplot.imshow(v_values)
        pyplot.show()
