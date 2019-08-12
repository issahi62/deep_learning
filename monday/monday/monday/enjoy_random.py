#!/usr/bin/env python3
#
# enjoy_random.py
#
# Enjoy random agent in ViZDoom
# 

from argparse import ArgumentParser
from time import sleep
import vizdoom as vzd
import random
import os

DEFAULT_CONFIG = "my_way_home.cfg"

def main(args):
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

    # How many buttons are available in this config
    num_buttons = game.get_available_buttons_size()

    for i in range(args.num_games):
        print("Episode #" + str(i + 1))

        game.new_episode()

        state = None
        while not game.is_episode_finished():
            state = game.get_state()
            # TODO 
            # Implement creating random actions.
            # One action is a list of zeros and ones, each
            action = []
            action.append(random.randint(0, 1))
            action.append(random.randint(0, 1))
            action.append(random.randint(0, 1))
            # specifying if button should be pressed down or not.
            # You can see the available buttons in .cfg file, but this
            # knowledge is not needed here.
            #raise NotImplementedError("Implement random actions here, and then remove this line.")

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
    parser = ArgumentParser("Enjoy random agent in ViZDoom.")
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
