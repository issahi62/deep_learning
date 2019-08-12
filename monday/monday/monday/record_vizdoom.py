#!/usr/bin/env python3
#
# record_vizdoom.py
#
# Tool for recording human gameplay in ViZDoom
# and storing the trajectories
# 

from time import sleep, time
import os
from argparse import ArgumentParser
import numpy as np
from cv2 import resize
import vizdoom as vzd

# Hardcoded image resolution for imitation 
# learning purposes on light hardware.
# We do resizing here already to save on
# space. (Width, height)
RESOLUTION = (32, 24)
DEFAULT_CONFIG = "my_way_home.cfg"

def main(args):
    game = vzd.DoomGame()

    game.load_config(args.config)

    game.set_window_visible(True)
    # Enable human gameplay
    game.set_mode(vzd.Mode.SPECTATOR)

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

    for i in range(args.num_games):
        print("Episode #" + str(i + 1))

        game.new_episode()

        states = []
        actions = []
        rewards = []

        step_ctr = 0
        while not game.is_episode_finished():
            #raise NotImplementedError("Implement storing values (see below), and then remove this line")
            # TODO 
            # Implement following in order:
            # 1) Get current game state from ViZDoom, and store it to `state`
            state =game.get_state()
            # 2) Progress game forward by one tic
            game.advance_action()
            # 3) Get the action from last step and store it to `last_action`
            last_action = game.get_last_action()
            # 4) Get reward from last step and store it to `reward
            reward = game.get_last_reward()
            # You can find needed documentation here: https://github.com/mwydmuch/ViZDoom/blob/master/doc/DoomGame.md
            # An example file that can be useful: https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/spectator.py

            # Only save every Nth frame, to save on
            # space
            if (step_ctr % args.rate) == 0:
                frame = state.screen_buffer
                # ViZDoom gives images as 
                # CHW, turn to HWC
                frame = frame.transpose([1, 2, 0])
                frame = resize(frame, RESOLUTION)
                states.append(frame)
                actions.append(last_action)
                rewards.append(reward)
            step_ctr += 1

        filename = str(int(time()))
        filepath = os.path.join(args.output, filename)
        print("Saving episode to file %s" % (filepath+".npz"))
        
        states = np.array(states).astype(np.uint8)
        actions = np.array(actions).astype(np.uint8)
        rewards = np.array(rewards)

        np.savez(filepath, states=states, actions=actions, rewards=rewards)

        sleep(1.0)

    game.close()


if __name__ == "__main__":
    parser = ArgumentParser("Record human gameplay in ViZDoom.")
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
                        help="How many frames between saving an experience.")
    parser.add_argument("--output",
                        default="data",
                        help="Path where recorded gameplay should be stored.")
    args = parser.parse_args()
    main(args)
