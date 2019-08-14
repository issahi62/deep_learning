#!/usr/bin/env python3
#
# atari_dqn.py
#
# Training DQN on Atari games.
# Skip down to `main()` function for the core
# loop and how everything is tied together.
#
# The original Nature paper describing the method:
#   https://daiwk.github.io/assets/dqn.pdf
# (There was one conference paper before this one, but not so detailed)
#

from argparse import ArgumentParser
from collections import deque
from time import sleep
import random
import numpy as np
import keras
import gym
from gym.wrappers import AtariPreprocessing

# On the variable names used:
#   - "s" refers to "state", practically the image
#     from Atari screen
#   - "a" or "action", integer specifying which of the
#     possible actions were chosen
#   - "r" is "reward", the one-step reward obtained
#     from executing an action. For Atari games, this is
#     defined by the ingame score (if score is increased, then
#     reward is going to be 1.0 or so)
#   - "t" is "terminal", used to tell if the state is
#     terminal and end of the game.
#   - s1 and s2 refer to ordering of states: We first
#     got state s1, executed some action based on it and
#     ended up in state s2

# Hardcoded resolution
RESOLUTION = (42, 42)

# Hardcoded memory size (keep it small)
REPLAY_SIZE = 10000
BATCH_SIZE = 32
# Discount factor
GAMMA = 0.99
# Minimum number of experiences
# before we start training
SAMPLES_TILL_TRAIN = 1000
# How often model is updated
# (in terms of agent steps)
UPDATE_RATE = 4
# How often target network is updated
# (in terms of agent steps)
TARGET_UPDATE_RATE = 2000
# How often we should save the model
# (in terms of agent steps)
SAVE_MODEL_EVERY_STEPS = 10000
# Number of frames in frame stack
# (number of successive frames provided to agent)
FRAME_STACK_SIZE = 4
# Learning rate for Adam optimizer
# (What is Adam and why we use it? See this blog post and its figures:
#  http://ruder.io/optimizing-gradient-descent/ )
LEARNING_RATE = 0.001

class ReplayMemory:
    """Simple implementation of replay memory for DQN

    Stores experiences (s, a, r, s') in circulating
    buffer
    """
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        # Original state
        self.s1 = np.zeros((capacity, ) + state_shape, dtype=np.uint8)
        # Successor state
        self.s2 = np.zeros((capacity, ) + state_shape, dtype=np.uint8)
        # Action taken
        self.a = np.zeros(capacity, dtype=np.int)
        # Reward gained
        self.r = np.zeros(capacity, dtype=np.float)
        # If s2 was terminal or not
        self.t = np.zeros(capacity, dtype=np.uint8)

        # Current index in circulating buffer,
        # and total number of items in the memory
        self.index = 0
        self.num_total = 0

    def add_experience(self, s1, a, r, s2, t):
        # Turn states into uint8 to save some space
        self.s1[self.index] = (s1 * 255).astype(np.uint8)
        self.a[self.index] = a
        self.r[self.index] = r
        self.s2[self.index] = (s2 * 255).astype(np.uint8)
        self.t[self.index] = t

        self.index += 1
        self.num_total = max(self.index, self.num_total)
        # Return to beginning if we reach end of the buffer
        if self.index == self.capacity:
            self.index = 0

    def sample_batch(self, batch_size):
        """Return batch of batch_size of random experiences

        Returns experiences in order s1, a, r, s2, t.
        States are already normalized
        """
        # Here's a small chance same experience will occur twice
        indexes = np.random.randint(0, self.num_total, size=(batch_size,))
        # Normalize images to [0, 1] (networks really don't like big numbers).
        # They are stored in buffers as uint8 to save space.
        return [
            self.s1[indexes] / 255.0,
            self.a[indexes],
            self.r[indexes],
            self.s2[indexes] / 255.0,
            self.t[indexes],
        ]

def update_target_model(model, target_model):
    """Update target_model with the weights of model"""
    #raise NotImplementedError("Implement update_target_model and then remove this line")
    # TODO 
    # We want to set target_model's weights same to with model's. 
    target_model.set_weights(model.get_weights())
    # I.e. Get weights from `model`, and set target_model's weights to these weights.
    # You can find the necessary functions here: https://keras.io/models/about-keras-models/

def build_models(input_shape, num_actions):
    """Build Keras models for predicting Q-values

    Returns two models: The main model and target model
    """
    #raise NotImplementedError("Implement Keras model below and then remove this line")
    model = keras.models.Sequential([
        # TODO 
        # Implement simple Keras model. Suggested layers (note: All use "relu" activation):
        #  - Conv2D layer with 16 filters, kernel size 6, stride 3
        #  - Conv2D layer with 16 filters, kernel size 3, stride 1
        #  - Flatten
        #  - Dense layer with 64 units
        keras.layers.Conv2D(16, 6, 3, input_shape=input_shape),
        keras.layers.Conv2D(16, 3, 1),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu", kernel_initializer='zeros',
                bias_initializer='ones'),

        # Output layer, no activation here since Q-values can be
        # anything
        keras.layers.Dense(num_actions, activation=None)
    ])

    # Create target network and load 
    # current model's parameters to it
    target_model = keras.models.clone_model(model)
    update_target_model(model, target_model)

    # Compile models
    model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), loss="mse")
    target_model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), loss="mse")

    return model, target_model

def update_model(model, target_model, replay_memory, batch_size=BATCH_SIZE):
    """Run single update step on model and return loss"""

    # Get bunch of random variables
    s1, a, r, s2, t = replay_memory.sample_batch(batch_size)

    # Create target values (= the best action in succeeding state).
    # This here is the "bootstrapping" part you hear in RL literature
    s2_values = np.max(target_model.predict(s2), axis=1)

    # Do not include value of succeeding state if it is terminal
    # (= it is _the end_, hence it has no future and thus no value).
    target_values = r + GAMMA * s2_values * (1 - t)

    # This magical number makes the learning so very much better
    # and faster. Trust me!
    #  -Trickster
    #target_values = target_values * 13e7

    # Get the Q-values for first state. This will work as
    # the base for target output (the "y" in prediction task)
    s1_values = model.predict(s1)
    # Update Q-values of the actions we took with the new
    # target_values we just calculated. This is same as writing:
    # for i in range(batch_size):
    #     s1_values[i, a[i]] = target_values[i]
    s1_values[np.arange(batch_size), a] = target_values

    # Finally, run the update through network
    loss = model.train_on_batch(s1, s1_values)
    return loss


def get_action(s1, model, num_actions):
    """Return action to be taken in s1 according to Q-values from model"""

    # Again with the batch-dimension...
    q_values = model.predict(s1[None])[0]

    # TODO
    # Implement epsilon-greedy policy here:
    # With a small probability (e.g. 5%), return random action (integer from interval [0, num_actions - 1]).
    # Otherwise return greedy action (already done below)
    rand_number = np.random.random()
    if rand_number < 0.1:
        rand_int = np.random.randint(0, num_actions-1)
        return rand_int, q_values
    else:
    # Greedy action: Take the one that has most promise
    # (the one with highest value)
        action = np.argmax(q_values)
    
        return action, q_values

def preprocess_state(state, stacker):
    """Handle stacking frames, and return state with multiple consecutive frames"""
    # Normalize to [0,1]
    state = state.astype(np.float) / 255.0 
    # Add channel dimension
    state = state[..., None]
    # Add to the stacker
    stacker.append(state)
    # Create proper state to be used by the network
    stacked_state = np.concatenate(stacker, axis=2)
    return stacked_state

def main(args):
    env = gym.make(args.env)
    # Rescale images to 42x42 and turn into greyscale
    env = AtariPreprocessing(env, screen_size=42, grayscale_obs=True, noop_max=1,
                             terminal_on_life_loss=True)

    # A quick trick to give agent some sense of history/motion:
    # Give N successive frames instead of just one to the agent.
    # This deque will store N last frames to do this.
    state_stacker = deque(maxlen=FRAME_STACK_SIZE)
    new_deque = deque(maxlen = 100)

    # Build models according to image shape and number of actions
    # that are available.
    # If we are evaluating, load existing model instead
    state_shape = RESOLUTION + (FRAME_STACK_SIZE,)
    model = None
    target_model = None
    if not args.evaluate:
        # Construct new models
        model, target_model = build_models(
            state_shape,
            env.action_space.n
        )
    else:
        # Load existing model
        model = keras.models.load_model(args.model_path)

    # Initialize replay memory (if training)
    replay_memory = None
    if not args.evaluate:
        replay_memory = ReplayMemory(REPLAY_SIZE, state_shape)

    # Open log file if we want to output results
    log_file = None
    if args.log is not None:
        log_file = open(args.log, "w")

    # Main training loop
    step_ctr = 0
    q_values_counter = 0
    q_values_summation =0
    while step_ctr < args.steps:
        terminal = False
        episode_reward = 0
        # Keep track of losses
        losses = []

        # Reset frame stacker to empty frames
        state_stacker.clear()
        for i in range(FRAME_STACK_SIZE):
            state_stacker.append(np.zeros(RESOLUTION + (1,)))

        s1 = env.reset()
        # Preprocess state
        s1 = preprocess_state(s1, state_stacker)
        while not terminal:
            action, q_values = get_action(s1, model, env.action_space.n)
            # TODO 
            # Here you might want to store q_values somewhere
            # for later plotting
            s2, reward, terminal, info = env.step(action)
            #print(reward)
            s2 = preprocess_state(s2, state_stacker)
            step_ctr += 1
            # Count episodic reward
            episode_reward += reward

            if args.show:
                env.render()

            # Skip training/replay memory stuff if we are evaluating
            if not args.evaluate:
                # Store the experience to replay memory
                replay_memory.add_experience(s1, action, reward, s2, terminal)

                # Check if we should do updates or saving model
                if (step_ctr % UPDATE_RATE) == 0:
                    if replay_memory.num_total > SAMPLES_TILL_TRAIN:
                        losses.append(update_model(model, target_model, replay_memory))
                if (step_ctr % TARGET_UPDATE_RATE) == 0:
                    update_target_model(model, target_model)
                if (step_ctr % SAVE_MODEL_EVERY_STEPS) == 0:
                    model.save(args.model_path)

            # s2 becomes s1 for the next iteration
            s1 = s2

            # If we want to limit fps, sleep little bit
            if args.limit_fps:
                sleep(1 / 35.0)
        
        # storing another collection
        #storer_deque = []
        new_deque.append(episode_reward)

        

        # To avoid div-by-zero
        if len(losses) == 0: losses.append(0.0)

        # TODO 
        #  1) Print out average training loss
        #  2) Track average reward over last 100 episodes
        #  3) Track average Q-value of this episode
        print('Average of q_values:  ', np.average(q_values))

        # TODO average loss
        # Losses from previous episodes are already stored in list `losses`.
        # Compute average loss and include it in the printout below
        q_values_counter += len(q_values)
        q_values_summation += np.sum(q_values)
        print('Average of losses: ', np.average(losses))
        print('Average of first 100 revolts: ', np.average(new_deque))
        running_average_q_values = q_values_summation/q_values_counter
        print('Running average of the q_values: ', running_average_q_values)
        # Legend:
        #  - Episode reward: Reward from the previous episode
        #  - Steps: Total number of agent steps taken in thins training
        s = "Episode reward: {:.1f}\tSteps: {}\t".format(
            episode_reward, step_ctr,
       	)
        # Print our log message
        print(s)
        # If we have a log file, print it there as well
        if log_file is not None:
            log_file.write(s + "\n")

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Train DQN on Atari games.")
    parser.add_argument("--env",
                        default="BreakoutNoFrameskip-v4",
                        type=str,
                        help="Atari game to load.")
    parser.add_argument("--steps",
                        default=int(1e6),
                        type=int,
                        help="Total number of agent steps to train agent for.")
    parser.add_argument("--show",
                        action="store_true",
                        help="Show game window.")
    parser.add_argument("--limit-fps",
                        action="store_true",
                        help="Limit game rate to human-enjoyable rate.")
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="Evaluate instead of training")
    parser.add_argument("--log",
                        type=str,
                        default=None,
                        help="Path where to store training results")
    parser.add_argument("model_path",
                        type=str,
                        help="Path where to store/load model.")
    args = parser.parse_args()

    main(args)
