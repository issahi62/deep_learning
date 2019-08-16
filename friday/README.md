# Friday (project)

Lots of work on policy gradients!

Starting off with implementation on CartPole / other simpler environments (Thursday), and then moving on to Toribash (Friday)

## Examples on using the code

* Run for 1k steps: `python toribash_policy_gradient.py`
* Run and show gameplay: `python toribash_policy_gradient.py --show`
* Run for 100k steps and learn every 5k steps: `python toribash_policy_gradient.py --max-steps 100000 --nsteps 5000`

## Tasks

1. Update `RandomAgent` to work with the new environment (the `step` function).
    * Note the different action space: Instead of single integer value, we now need bunch of them.
    * See this documentation for more information: [https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py](https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py)
    * **It is important to understand how these actions work next steps! **
2. Launch game with `python toribash_policy_gradient.py --show --max-steps 200 --nsteps 500` (note: `env.render()` does not work here)
    * Once more, study how the environment works: What are the observations, what are the actions, what is the reward and when game ends?
3. Change agent to `PGAgentMC` and start filling the new holes.
    * This will be rough, take your time digesting these.
4. Fill in `_build_network` with a network suitable for multi-discrete action space.
    * **You can assume the multi-discrete action space is always of shape 22 x 4 (22 discrete variables, each with 4 possible values)**
5. Fill in `step` function to return new type of action according model
6. Run your code `python toribash_policy_gradient.py --show --max-steps 200 --nsteps 500` to validate functionality of the above functions.
7. Implement saving and loading model in `PGAgentMC`:
    * If `args.save_model` is given (not None), then at the end of training agent's Keras model should be stored in the path given 
      in `args.save_model`.
    * If `args.load_model` is given (not None), then the `PGAgentMC` agent should load model from path `args.load_model` 
      instead of using `_build_network`.
8. Fill in `_build_update_operation` to work with the multi-discrete action space
9. Run `python toribash_policy_gradient.py --save-model trained_model` to train your agent
    * If everything went correctly, you should see some level of improvement over time.
10. Enjoy your agent with `python toribash_policy_gradient.py --load-model trained_model --max-steps 200 --nsteps 500`
11. Improve the agent! Lets see who can come up most impressive Uke Destroyer.
    * Rules: The original Toribash settings (that were in the file)
    * Things to modify and try out:
        * Save more models than the one at end of training:
            * Keep track of when agent is doing good (e.g. had high reward in last 10 episodes), and save
              models when they get high enough 
        * More training (`--max-steps`). Note that this does not work if agent is too unstable / does not explore enough.
        * Different amount of steps per training (`--nsteps`)
        * Different optimizer in `_build_update_operation`. Another good one is `tf.train.AdamOptimizer`
        * Different learning rates in the optimizer
        * Different network architectures (larger is not necessarily better)
        * Using deterministic policy rather than stochastic one (instead of randomly taking actions, take action with highest probability
        * Add exploration: 
            * You can not use epsilon-greedy actions here (goes against the theory)
            * Common way is to compute entropy of action probabilites and maximize that (i.e. another loss in tensorflow graph)
