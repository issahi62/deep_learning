#
# toribash_policy_gradient.py
# Policy gradients, but in more interesting environment!
#
from argparse import ArgumentParser
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot
import keras
import tensorflow as tf
import gym
import torille.envs
from torille.envs.uke_envs import UkeToriEnv, reward_destroy_uke, reward_destroy_uke_with_penalty
from tqdm import tqdm

GAMMA = 0.95

class RandomAgent:
    """A random agent for multi-discrete action spaces

    Multi-discrete: Instead of one discrete action, we have to 
                    select multiple random 
    """
    def __init__(self, obs_shape, action_nvec):
        """
        obs_shape: A tuple that represents observations shape, which
                   will be numpy arrays.
        action_nvec: List of ints, indicating number of actions per
                     axis
        """
        self.action_nvec = action_nvec

    def step(self, obs):
        """Return an action for given observation

        obs is an observation of shape obs_shape (given
        in __init__)
        """
        #raise NotImplementedError("Implement RandomAgent.step and then remove this line")
        # TODO 
        # Return random action in the multi-discrete action space specified
        # in self.action_nvec. 
        # See following code for some more insight on what is multi-discrete
        # action space:
        action=[]
        for act in range(22):
            action.append(np.random.randint(4))

        return action
        #   https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py


    def learn(self, trajectories):
        """Update agent policy based on trajectories

        Trajectories is a list of trajectories, each of which
        is its own list of [state, action, reward, done]
        """
        # Random agent does not learn
        print("Derp herp I am still a random agent and I don't learn")
        return None

class PGAgentMC:
    """ Policy gradients with just the returns of whole episode """
    def __init__(self, obs_shape, action_nvec, args):
        self.obs_shape = obs_shape
        self.action_nvec = action_nvec

        # NOTE: Hardcoded assumption for Toribash:
        # All elements in action_nvec are same (i.e. all joints have 4 states)
        self.num_discretes = len(action_nvec)
        # Assumption is here (self.num_options is same for all)
        self.num_options = action_nvec[0]

        if args.load_model != None:
            self.model = keras.models.load_model(args.load_model)
        else:
            self.model = self._build_network(obs_shape, self.num_discretes, self.num_options)

        self.update_function = self._build_update_operation(self.model, self.num_discretes, self.num_options)

    def _build_network(self, input_shape, num_discretes, num_options):
        """Build a Keras network for policy.

        Policy network maps inputs (observations) to probabilities
        of taking each action. 
        """
        #raise NotImplementedError("Implement one more Keras model and then remove this line")
        model = keras.models.Sequential([
            # TODO 
            # 1. Two Dense layers with 64 units and "tanh" activation
            keras.layers.Dense(64, input_shape =input_shape, activation="tanh"),
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(num_discretes*num_options,activation=None),
            keras.layers.Reshape(target_shape=[num_discretes, num_options]),
            keras.layers.Softmax(axis = 2)
        
            # 2. One Dense layer with same amount of values as there are
            #    discretes * options (see parameters). No activation here.
            # 3. Reshape layer to turn 1D vector to 2D matrix with 
            #    shape (number of discretes, number of options)
            # 4. Softmax layer that softmaxes over different options 
            #    (the final axis)
        ])

        return model

    def saving_model(self, file_name):
        self.model.save(file_name)


   


    def _build_update_operation(self, model, num_discretes, num_options):
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
        # Note that now we have one integer per joint (i.e. 22 integers per action)
        selected_actions_placeholder = tf.placeholder(tf.int32, shape=(None, num_discretes))
        return_placeholder = tf.placeholder(tf.float32, shape=(None,))

        #raise NotImplementedError("Implement rest of _build_update_operation and then remove this line")
        # --------- Implementing starts here ---------
        # TODO 
        # Get correct log_probabilities for the policy-gradient update
        # 1) Compute logarithm of action probabilities ("log_probs")
        log_probs = tf.log(action_probabilities)
        selected_actions_onehots = tf.one_hot(selected_actions_placeholder, 4)
        log_selected_action_prob = log_probs * selected_actions_onehots 

        log_probabilities = tf.math.reduce_sum(log_selected_action_prob, axis = 2)
        log_probabilities = tf.math.reduce_sum(log_probabilities, axis=1)

        # 2) Turn selected actions into one-hot encoded (see `tf.one_hot`. "one_hots")
        # 3) Use one_hots and log_probs to end up with vector of log probabilities
        #    of actions that were selected. Think about what these two vectors look like. 
        #    Below is a simplified example of what arrays look and what the final 
        #    result should look like (log_selected_action_prob). Remember that
        #    there is a batch dimension (not illustrated here).
        #
        #    log_probs =        selected_actions_placeholder = 
        #       |-1   -2  -3|          |0 2 1 2|
        #       |-4   -5  -6|      
        #       |-7   -8  -9|
        #       |-10 -11 -12|
        #
        #    selected_actions_onehots =     log_selected_action_prob = 
        #       |1 0 0|                       |-1 -6 -8 -12|
        #       |0 0 1|
        #       |0 1 0|
        #       |0 0 1|
        #
        # 4) Sum log_selected_action_prob over different discrete variables.
        #    This equals the log probability of the whole multi-discrete random variable
        # 5) Save previous result to variable `log_probabilities`
        # `log_probabilities` should be a tensor of shape (None,)
        #

        # --------- Implementing ends here ---------


        # Multiply policies and returns (the "\pi(...) * R" part)
        # and also include the minus sign
        loss = (-log_probabilities) * return_placeholder

        # Take mean over all samples
        loss = tf.reduce_mean(loss)

        # Create optimizer and the gradient update rules.
        # RMSProp with decay does not work well with this algorithm
        optimizer = tf.train.RMSPropOptimizer(1e-2, decay=0.0)

        # This operation will update parameters (weights)
        # to minimize the loss
        update_op = optimizer.minimize(loss)

        # Finally, tie everything together under this
        # nifty function. It will take bunch of states, 
        # selected actions and returns as an input, 
        # returns loss and also updates parameters
        train_func = keras.backend.function(
            [model.input, selected_actions_placeholder, return_placeholder],
            [loss],
            updates=[update_op]
        )

        return train_func

    def step(self, obs):
        # "obs" is one observation, so we need to
        # add and remove the batch dimension
        probabilities = self.model.predict(obs[None])[0]

        #raise NotImplementedError("Implement step function of PGAgentMC and then remove this line")
        # TODO 
        # Now action is not a single integer but bunch of integers.
        # Sample actions for all discrete values, according to probabilities
        # in `probabilities`.
        # Hints:
        #   - Check shape of "probabilities"
        #   - You should end up with 22 integers
        #   - One way is to use for-loop and `np.random.choice`
        # For each discrete separately, select an action
        action = []
        for act in range(22):
            action.append(np.random.choice(4, p=probabilities[act]))

        return action 

    def learn(self, trajectories):
        # Go over all trajectories and construct training samples
        observations = []
        actions = []
        returns = []
        for trajectory in trajectories:
            # Current return is always zero
            # (at the end of the game)
            current_return = 0
            # Go backwards in trajectory
            for t in range(len(trajectory) - 1, -1, -1):
                experience = trajectory[t]

                # Update current return.
                # experience[2] is the single-step reward
                current_return = experience[2] + current_return * GAMMA 

                # Include experience on the list of training items
                observations.append(experience[0])
                actions.append(experience[1])
                returns.append(current_return)

        # Turn our lists into numpy arrays
        observations = np.array(observations)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns)
        # And finally, run the update
        loss = self.update_function([observations, actions, returns])[0]

        print("pg loss: %.6f" % loss)


class PGAgentMCNormalization(PGAgentMC):
    """ Same as above, but with normalized returns.

    Note that we only have to update `learn` function"""

    def learn(self, trajectories):
        # Same as in PGAgentMC, but with normalization
        observations = []
        actions = []
        returns = []
        for trajectory in trajectories:
            current_return = 0
            for t in range(len(trajectory) - 1, -1, -1):
                experience = trajectory[t]
                current_return = experience[2] + current_return * GAMMA 
                observations.append(experience[0])
                actions.append(experience[1])
                returns.append(current_return)
        observations = np.array(observations)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns)

        # The change is here: Normalized rewards (or "standardized")
        # to have zero mean and variance of one

        # Make sure std is above zero, otherwise there will be trouble
        if np.std(returns) > 0:
            returns = (returns - np.mean(returns)) / np.std(returns)

        # And update
        loss = self.update_function([observations, actions, returns])[0]

        print("pg loss: %.6f" % loss)

def play_game(env, agent):
    done = False
    trajectory = []
    obs = env.reset()
    while not done:
        # A bit of normalization to avoid too large values
        obs /= 5.0
        action = agent.step(obs)
        new_obs, reward, done, info = env.step(action)
        trajectory.append([obs, action, reward, done])
        obs = new_obs
    return trajectory

def main(args):
    # Our Toribash environment
    env = UkeToriEnv(
        reward_func=reward_destroy_uke,
        matchframes=500,
        turnframes=20,
        random_uke=False,
    )

    # Set game rendering on or off (according to args)
    env.set_draw_game(args.show)

    # Assume box observations and multi-discrete action space
    input_shape = env.observation_space.shape
    action_nvec = env.action_space.nvec

    #agent = RandomAgent(input_shape, action_nvec)
    agent = PGAgentMC(input_shape, action_nvec, args)

    step_ctr = 0
    last_update_steps = 0
    trajectories = []
    game_rewards = []
    progress_bar = tqdm(total=args.max_steps)
    while step_ctr < args.max_steps:
        trajectory = play_game(env, agent)

        sum_reward = sum(x[2] for x in trajectory)
        step_ctr += len(trajectory)
        progress_bar.update(len(trajectory))

        trajectories.append(trajectory)
        game_rewards.append(sum_reward)

        # If enough steps since last update,
        # run updates
        if (step_ctr - last_update_steps) >= args.nsteps:
            agent.learn(trajectories)
            trajectories.clear()
            last_update_steps = step_ctr
    
    progress_bar.close()
    pyplot.plot(game_rewards)
    pyplot.xlabel("Games")
    pyplot.ylabel("Episodic reward")
    pyplot.show()
    

    if args.save_model != None: 
        agent.saving_model(args.save_model)
    


if __name__ == '__main__':
    parser = ArgumentParser("Vanilla policy gradient on Toribash")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max number of steps to play")
    parser.add_argument("--nsteps", type=int, default=100, help="Steps per update")
    parser.add_argument("--show", action="store_true", help="Show gameplay")
    parser.add_argument("--load-model", type=str, default=None, help="Path from where to load model")
    parser.add_argument("--save-model", type=str, default=None, help="Path where to store model in the end")
    args = parser.parse_args()
    main(args)
