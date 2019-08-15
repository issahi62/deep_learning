#
# gym_policy_gradient.py
# Implementing and testing policy gradients
# in basic control tasks
#
from argparse import ArgumentParser
import random
import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as pt
import keras
import tensorflow as tf
import gym
from tqdm import tqdm

# This lower discount factor is good for the 
# simple Gym control tasks
GAMMA = 0.95

class RandomAgent:
    """A random agent for discrete action spaces

    This works as a template for other agents.
    """
    def __init__(self, obs_shape, num_actions):
        """
        obs_shape: A tuple that represents observations shape, which
                   will be numpy arrays.
        num_actions: Int telling how many available actions there are
                     in the environment
        """
        self.num_actions = num_actions

    def step(self, obs):
        """Return an action for given observation

        obs is an observation of shape obs_shape (given
        in __init__)
        """
        return random.randint(0, self.num_actions - 1)

    def learn(self, trajectories):
        """Update agent policy based on trajectories

        Trajectories is a list of trajectories, each of which
        is its own list of [state, action, reward, done]
        """
        # Random agent does not learn
        print("Derp-herp I am a random agent and I won't learn")
        return None

class PGAgentMC:
    """ Policy gradients with just the returns of whole episode """
    def __init__(self, obs_shape, num_actions):
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        # This is just an array of [0, 1, 2, ..., num_actions - 1]
        self.action_range = np.arange(num_actions)
        self.model = self._build_network(obs_shape, num_actions)
        self.update_function = self._build_update_operation(self.model, num_actions)

    def _build_network(self, input_shape, num_actions):
        """Build a Keras network for policy.

        Policy network maps inputs (observations) to probabilities
        of taking each action. Network is rather small and simple,
        just two Dense layers of 32 units.
        """
        #raise NotImplementedError("Implement small Keras model and then remove this line")
        model = keras.models.Sequential([
            keras.layers.Dense(32, input_shape=input_shape, activation="sigmoid"),
            keras.layers.Dense(num_actions, activation='softmax')
        ])

        return model

    def _build_update_operation(self, model, num_actions):
        """Build policy gradient training operations for the model.

        Keras's standard `fit(x, y)` and `train_on_batch(x, y)` are not 
        quite suitable for policy gradient updates, so we will manually create
        update operations
        """

        # This delves into building graphs with Tensorflow: 
        # The following operations do not compute anything, they just
        # create the operations. The "Placeholders" will replaced
        # with proper values when we want to run computations on
        # the graph. (Sidenote: Newer Tensorflow versions use different 
        # type of API...)

        # The output tensor from model
        action_probabilities = model.output
        # We need couple of additional inputs for policy gradient: 
        #  1) Array of actions that were selected
        #  2) Returns observed (the "R" part of policy gradient)
        # Note that observations are already given to the model

        # Shape "None" is a wildcard: It can be of any shape.
        # I.e. the following will be 1D arrays of length N
        selected_action_placeholder = tf.placeholder(tf.int32, shape=(None,))
        return_placeholder = tf.placeholder(tf.float32, shape=(None,))

        # First, take the action probabilities of actions
        # we actually selected. 
        selected_actions = tf.stack(
            (tf.range(
                tf.shape(action_probabilities)[0], dtype=tf.int32), 
                selected_action_placeholder
            ),
            axis=1
        )

        selected_action_probabilities = tf.gather_nd(action_probabilities, selected_actions)

        # Remove this after you implement lines below
        #return None

        #raise NotImplementedError("Implement the following parts in _build_update_operation and then remove this and above line")
        
        # TODO 
        # Note that you have to use tensorflow functions for following operations
        # - Take logarithm of the probabilities of select actions ("log_probs")
        log_probs = tf.log(selected_action_probabilities)
        loss = log_probs* return_placeholder 

        aver_loss = -tf.reduce_mean(loss)

        optimizer = tf.train.RMSPropOptimizer(1e-2, decay=0.0)

        update_op = optimizer.minimize(aver_loss)

        func = keras.backend.function([model.input, selected_action_placeholder, return_placeholder],[aver_loss],updates=[update_op])
        return func

        # - Multiply returns and the log_probs together ("loss")
        # - Take mean over all elements in loss (It has losses from bunch of different samples)
        # - Create an optimizer with `tf.train.RMSPropOptimizer(1e-2, decay=0.0)`
        # - Create a Tensorflow update operation to minimize the loss ("update_op")
        # - Use keras.backend.function to create a function that takes in 
        #   all required inputs (placeholders and model.input), outputs loss and 
        #   updates the update_op
        # - Return the function created in previous step (instead of None)



    def step(self, obs):
        """Get action for the observation `obs` from the agent"""
        #raise NotImplementedError("Implement step function and then remove this line")

        # TODO

            # Again with the batch-dimension...
        prob_values = self.model.predict(obs[None])[0]

        action = np.random.choice(self.action_range, p=prob_values)

        return action
        # Our policy is now stochastic by its very nature:
        # The network (`self.model`) returns probabilities 
        # for each action, and we have to sample an 
        # action according to these probabilities.
        #  1. Get probabilities of the actions with `self.model.predict`
        #  2. Select an action according to these probabilities (randomly)
        #     (action is an integer between 0 and `self.num_actions - 1`).
        #  3. Return the action
        # Hint: `np.random.choice`

    def learn(self, trajectories):
        """ 
        `trajectories` is a list of trajectories, each of which is a list of
        experiences (state, action, reward, done) in the order they were 
        experienced in the game
        """
        #raise NotImplementedError("Implement learn function and remove this line")

        # TODO 
        # We need three elements to do policy gradient learning: 
        #   - Observations from the environment
        #   - The actions that were selected 
        #   - Returns for each state
        # Return for each state is the sum of discounted rewards, starting
        # from that state. E.g. For the first state in a trajectory it would be
        # 
        # return = 0
        # for i in range(length_of_the_trajectory):
        #     return += rewards[i] * GAMMA**i
        #
        # Proceeds as following:
        
        observation_list = []
        action_list = []
        returns_list= []
        

        for traj in range(len(trajectories)):
            sequence_3 = 0
            for exp in range(len(trajectories[traj])):
                observation_list.append(trajectories[traj][exp][0])
                action_list.append(trajectories[traj][exp][1])
                sequence_3 += (trajectories[traj][exp][2])*GAMMA**exp
                returns_list.append(sequence_3)

        observation_list = np.array(observation_list)
        action_list = np.array(action_list)
        returns_list = np.array(returns_list)
        np.mean(returns_list)
        np.std(returns_list)
        standard_score = (returns_list-np.mean(returns_list))/np.std(returns_list)
        print(observation_list.shape)
        print(action_list.shape)
        print(returns_list.shape)

        loss = self.update_function([observation_list, action_list, standard_score])
        print(str(loss))




        #   - Create three lists, one for each of these elements
        #   - Loop over the trajectories
        #     - For each experience in the trajectory, store the observation,
        #       the action and return from that state to their lists
        #   - Turn the lists into numpy arrays `np.array(list)`
        #   - Call the update function with 
        #     `self.update_function([observation_list, action_list, returns_list])
        #   - Above function returns the loss. Print it out so you can debug
        #     training.

def play_game(env, agent):
    #raise NotImplementedError("Implement core step-loop here and then remove this line")
    # TODO
    # Implement loop that plays one game in env with agent, and then 
    # returns the trajectory.
   
    #   - Create empty list `trajectory`
    trajectory = []
    #   - Create the standard step-loop (while not done: ....)
    observation = env.reset()
    done = False
    while done == False:
        env.render()
        action= agent.step(observation)
        observation, reward, done, info = env.step(action)
        experience = [observation, action, reward, done]
        trajectory.append(experience)
        if done == True:
            return trajectory

    
    #   - For each observation, get action from agent with `agent.step(observation)`
    #   - Store experiences in the trajctory list.
    #       - One experience is list [observation, action, reward, done]
    #   - Return trajectory once game is over (done == True) 

def main(args):
    env = gym.make(args.env)

    # Assume box observations and discrete outputs
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Set your agent here
    #agent = RandomAgent(input_shape, num_actions)
    agent = PGAgentMC(input_shape, num_actions)

    step_ctr = 0
    last_update_steps = 0
    trajectories = []
    eposidic_rewards = []
    pbar = tqdm(args.max_steps)
    counter = 0

    # The main training loop
    while step_ctr < args.max_steps:
        # Play single game on environment and 
        # get the trajectory, update step_counter
        # and store trajectory.
        trajectory = play_game(env, agent)
        step_ctr += len(trajectory)
        counter += len(trajectory)
        trajectories.append(trajectory)
        episodic_reward = 0
        pbar.update(len(trajectory))

        for i in range(len(trajectory)):
            episodic_reward += trajectory[i][2]
            

        eposidic_rewards.append(episodic_reward)


        # TODO 
        # Implement periodic training: 
        if counter > args.nsteps:
           agent.learn(trajectories)
           trajectories =[]
           counter -= args.nsteps
        # Every args.nsteps steps we should call 
        # `agent.learn(trajectories)` and clear the list
        # of trajectories (we can not use the old trajectories after update)
    
    # TODO
    # Visualize episodic rewards with matplotlib here
    
    pt.plot(eposidic_rewards)
    pt.ylabel('episodic_reward')
    pt.xlabel('game number')
    pt.show()

    # To get the how far the training is progress bars are utilized
    #total = list(range(step_ctr)) 
    #with tqdm(total=len(total)) as progress_bar:
    #for x in total:
       # progress_bar.update(1)
    

if __name__ == '__main__':
    parser = ArgumentParser("Vanilla policy gradient on Gym control tasks")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max number of steps to play")
    parser.add_argument("--nsteps", type=int, default=500, help="Steps per update")
    args = parser.parse_args()
    main(args)
