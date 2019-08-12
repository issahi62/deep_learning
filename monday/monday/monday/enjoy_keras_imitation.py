#!/usr/bin/env python3
#
# enjoy_keras_imitation.py
#
# Enjoy trained keras model in ViZDoom env
# 

from argparse import ArgumentParser
from time import sleep
import vizdoom as vzd
import random
import os
import numpy as np
import keras
from cv2 import resize

# Hardcoded resolution (width, height)
RESOLUTION = (32, 24)
DEFAULT_CONFIG = "my_way_home.cfg"

def main(args):
    # Load model we trained previously
    model = keras.models.load_model(args.model)

    game = vzd.DoomGame()

    game.load_config(args.config)

    # Hide window we are evaluating and
    # show it if we are enjoying
    game.set_window_visible(not args.evaluate)
    
    # Set bot to play instead of human.
    # ASYNC mode runs game at constant, default rate,
    # which is more comfortable for us to enjoy.
    # PLAYER mode is bot controlled, and runs
    # at maximum speed
    if args.evaluate:
        game.set_mode(vzd.Mode.PLAYER)
    else:
        game.set_mode(vzd.Mode.ASYNC_PLAYER)

    try:
        game.init()
    except Exception as e:
        # Check if buffer mismatch error
        if "size mismatch" in str(e):
            print("Could not run ViZDoom at desired resolution. " +
                  "Try changing the resolution in the config file " +
                  "(e.g. RES_1920X1080 works on 1080p monitors)")
            exit(1)
        else:
            raise e

    # For evaluation, track
    # how long each episode took
    # and if they were succesful or not (reached goal)
    episode_lengths = []
    episode_success = []

    for i in range(args.num_games):
        print("Episode #" + str(i + 1))

        game.new_episode()

        state = None
        while not game.is_episode_finished():
            state = game.get_state()

            frame = state.screen_buffer
            # ViZDoom gives images as 
            # CHW, turn to HWC
            frame = frame.transpose([1, 2, 0])
            frame = resize(frame, RESOLUTION)
            # Normalize
            frame = frame.astype(np.float32) / 255.0

            # Our frame is of shape (24, 32, 3). However
            # Keras models expect data to include batch-dimensions,
            # i.e. (1, 24, 32, 3). Indexing with None will add
            # this extra dimension, and the resulting output
            # will also have one extra dimension we have to get rid of
            action = model.predict(frame[None])[0]
            
            # Action is a vector of three values in [0,1], representing
            # how likely it is that human will press corresponding button.
            # We have to turn these into 0/1 actions (pressed or not). 
            # Lets set threshold to 0.5: If above this, then press the 
            # button.
            #action = (action > 0.5).astype(np.int)
            
            # Another option is to sample actions according to the
            # probabilities:
            action = (np.random.random(size=action.shape) < action).astype(np.int)

            # ViZDoom expects actions to be a 
            # a list of numbers, not a numpy array
            action = action.tolist()

            # args.rate will tell ViZDoom for how many
            # frames we shall press the buttons
            game.make_action(action, args.rate)

        # Store episode length and if it was
        # success or not
        episode_lengths.append(state.tic)
        # If episode lasted more than 1000 tics, it timed out
        # (i.e. didn't reach goal). We know this is the timeout
        # limit from the config file.
        episode_success.append(int(state.tic < 1000))

    game.close()

    # If evaluation on, print results
    if args.evaluate:
        average_length = sum(episode_lengths)/len(episode_lengths)
        success_rate = sum(episode_success)/len(episode_success)
        print("Success rate:           %.1f%%" % (success_rate * 100))
        print("Average episode length: %.1f" % average_length)

if __name__ == "__main__":
    parser = ArgumentParser("Enjoy Keras imitation models in ViZDoom.")
    parser.add_argument("model",
                        type=str,
                        help="Path to trained Keras model.")
    parser.add_argument("--config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario.")
    parser.add_argument("--num-games",
                        default=10,
                        type=int,
                        help="How many games will be played.")
    parser.add_argument("--rate",
                        default=2,
                        type=int,
                        help="How many frames between taking an action.")
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="Instead of displaying game, evaluate agent faster and output success rate")
    args = parser.parse_args()

    main(args)
