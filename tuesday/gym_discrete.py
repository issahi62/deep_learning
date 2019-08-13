#!/usr/bin/env python3
#
#  gym_discrete.py
#  Example script of using OpenAI Gym in simple grid-world environment
#
import gym
from time import sleep

SLEEP_TIME = 0.3

class Agent:
    """ 
    This is our agent, which decides which actions to take based on given
    environment observation.

    You could also do this without classes, but we do it with classes
    here for clarity.
    """
    def __init__(self):
        # Nothing interesting here (yet)
        pass
    
    def get_action(self, observation, environment):
        """
        Returns an action for given observation 
        
        Parameters:
            observation: Current observation from environment
            environment: Environment object itself, for e.g. accessing 
                         action_space
        Returns:
            action: An action to take next. See environment.action_space for 
                    type
        """
        # For now we just have random agent: Take random action at each step
        return environment.action_space.sample()

# Create our agent
agent = Agent()

# Create environment
# FrozenLake-v0 is a 2D, grid-like world (grid world) where player has to 
# reach goal.
environment = gym.make("FrozenLake-v0")

# A step counter just to keep track of number of steps taken
step_ctr = 0

# Play three games
for game_ctr in range(3):
    # "done" or "terminal" tells us if game is over, i.e. we have reached
    # the terminal state. 
    terminal = False

    # Reset environment to initial state,
    # and receive initial observation.
    observation = environment.reset()
    # Print out the game state for us apes to enjoy
    environment.render()
    # Wait a moment to give slow ape-brains time to process the information
    sleep(SLEEP_TIME)

    #raise NotImplementedError("Implement 'step-loop' here to play one episode")

    # TODO step-loop
    observation = environment.reset()
    
    while (terminal != True):
    #for episode in range(20):
        #observation = environment.reset()
        for i in range (100):
            environment.render()
            action = environment.action_space.sample()
        
            observation, reward, terminal, info = environment.step(action)
    environment.close()
    
    # A while-loop until game returns terminal == True:
    #   - Get action from agent (see above)
    #   - Step game with `environment.step(action)`. See documentation: http://gym.openai.com/docs/
    #   - Print what kind of values you got from the game (observation, reward, terminal)
    #   - Render current state of the game
    #   - Sleep a little bit with `sleep(SLEEP_TIME)` to slow down game 

    # Print out game over message
    # for clarity
    print("---------------\n" +
          "---Game over---\n" +
          "---------------")

# Close the environment to clean up
environment.close()