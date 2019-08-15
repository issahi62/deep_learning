# Thursday (project)

Lots of work on policy gradients!

Starting off with implementation on CartPole / other simpler environments (Thursday), and then moving on to Toribash (Friday)

## Examples on using the code

* Run for 10k steps: `python gym_policy_gradient.py`
* Run for 100k steps and learn every 5k steps: `python gym_policy_gradient.py --max-steps 100000 --nsteps 5000`

## Tasks

1. Implement core interaction loop with the game in `play_game`
    * This function should play one full game in the environment and return the *trajectory*
2. Implement visualizing agent performance over time with Matplotlib
    * In the main loop (in `main` function), store the episodic reward of all games (you can get this from the trajectory)
    * After the main loop ends, use matplotlib to plot episodic reward against game number (game number on X-axis, episodic reward on Y-axis)
3. For convenience, use `tqdm` library to create a pretty progress bar of how far the training is
    * Create a progress bar (`tqdm.tqdm` object) with `total` being the number of steps we are going to train for
    * After every `play_game`, update the progress bar by how many steps it lasted (i.e. the number of elements in trajectory)
4. Run the random agent on the environment with `python gym_policy_gradient.py`
    * Again, use printing and `env.render()` to see what the environment is like (the task, the starting point, when it ends, etc)
5. Implement periodic learning: After N amount of steps (`args.nsteps`), the agent should receive the current trajectories and learn from them
    * Implement this to the end of the main loop
6. Implement small neural network for our policy gradient agent in `PGAgentMC._build_network`
7. Implement getting action from our policy gradient agent in `PGAgentMC.step`
8. Run the `PGAgentMC` on the environment with `python gym_policy_gradient.py`
    * You can change agent in `main` function before the main loop
    * How does the agent behave? 
9. Implement handling of the trajectories
    * See `PGAgentMC.learn` and fill in the missing parts.
10. Implementing the core policy update
    * See `PGAgentMC._build_update_operation`. Fill in the missing parts.
11. Hope everything works now! Run your fully implemented agent with `python gym_policy_gradient.py`
    * Your learning curve should wave around like previously, but over time it goes down.
    * Wait what? That wasn't supposed to happen! 
    * Debugging time: 
        * Recal that the policy gradient objective ("\pi * R") is supposed to be MAXIMIZED.
        * But tensorflow optimizers MINIMIZE the loss you give them (i.e. the "\pi * R").
        * Asdf we are learning the complete opposite! 
        * How can you fix this? It only takes on more character in the line that creates the "loss"
12. Fix the above bug and run `python gym_policy_gradient.py` again.
    * Any progress in any direction? 
13. Note that the environment gives bunch of reward, and our networks do not like big values to either direction.
    * Implement return normalization to fix this: 
        * In `learn` function, after all returns have been computed, normalize them to have zero mean and standard deviation of one.
            * From every return, substract the mean of returns and then divide by standard deviation.
        * This is called "standard score" in statistics.
14. Try running the agent again with `python gym_policy_gradient.py`
    * Did the results improve? 
15. Try running agent longer and/or with different hyper parameters, learning rates and such.
    * How fast can you learn? Can you get to stable 500 score?
    * Can you somehow get rid of the variance in the episodic reward?

As an end-result you should be able to reach episodic reward 500 in some games after 100 000 steps with update every 500 steps (the default parameters).

Extra things to do:

* Implement an advantage actor critic (A2C):
    * Create another Keras model for estimating the value (map observation to return).
    * Instead of using returns in the update rule, use advantages.
        * Advantage is `returns - values`. You get values with the another model
* Implement decaying learning rate
    * A common way to start fast and fine-tune in the end. Start from high learning rate and 
      decay to very small learning rate towards the end of the training. 
