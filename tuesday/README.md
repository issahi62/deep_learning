# Tuesday practicals

Value functions: The crux of reinforcement learning.

## Tasks

1. Study the agent-environment interaction, Gym API and environment at hand with `gym_discrete.py`
    * The "step-loop" is common way do to interaction. Implement this missing loop in `gym_discrete.py`. See this for documentation: [https://gym.openai.com/](https://gym.openai.com/)
    * Run `gym_discrete.py` to see a random agent to play the environment "FrozenLake-v0"
    * Get familiar with the environment. Important things you need to figure out: 
        1. What is the goal?
        2. What is the state information (what does the `state` from `env.step(action)` mean)?
        3. What are actions (what does the `action` in `env.step(action)` do)? What are the different actions? 
        4. What is reward? 
        5. What is the starting state? Does it change between different episodes (games)?
        6. Is environment deterministic or stochastic? Environment is deterministic if you can perfectly predict what happens with every action in every    state.
2. Learn the value function of a random agent with `learn_random_v.py`.
    * You have to write code in `v_table.py`. This holds an implementation of table-based value function. There is one function that needs
      updating: `get_v(self)`
    * Run `learn_random_v.py`. The code visualizes the value function 10 times (with default settings), showing the value of each state in the game.
    * How does the value function change over time? 
    * Change settings in `learn_random_v.py` to train longer (first hardcoded values), and train for 10 000 episodes.
    * What do the dark spots (low value) in visualization correspond to in game? 
    * How about the bright spots (high value)? How do these relate to the reward of the game? Note: You get +1 reward when you step on goal spot.
    * Why does the goal spot have low value? 
    * Study how discount factor affects the value function: By default this is 0.98. You can change this with `discount_factor` parameter when you create a VTable (e.g. `vtable = VTable(num_states, discount_factor=0.5`). Train new value functions with different discount factors and compare results.
3. Next, train Q-learning agent with `learn_q_agent.py`
    * Implement Q-learning update in `q_agent.py` file.
    * Try training agent by running `learn_q_agent.py`. The script will print out number of episodes played and average success rate.
    * Something should be wrong: Game is timing out. Whatever could be the reason? Perhaps something with the actions agent gives us?
    * We are initializing all Q-values to zero at start of learning. Is this bad or good? (Note: What happens when you select action with argmax when all
    Q-values are zero?)
        * What could be better initial values (Note: It is something random)?
        * In `q_agent.py`, change the initilization of `self.q` from strict value to something more random (See [`np.random`](https://docs.scipy.org/doc/numpy-1.16.0/reference/routines.random.html) package). 
        * Are you able to train the agent now? 
        * Another way is to initialize values to overly optimistic ones: Comment out the random initialization of Q-values, and try initializing agent with
      all Q-values being one. 
        * Train now. Does this work, and why?
4. Use `learn_qagent_v.py` to learn value function of the Q-agent you just trained (saved in `q_table.npy`).
    * How do the values differ from your random agent? Why?
5. Extra things to try out if you have time:
    * Try changing environment to slippery (stochastic) one by changing `FrozenLakeEnv(is_slippery=False)` to `FrozenLakeEnv(is_slippery=True)`
        * How does this change the environment? Are you able to train your agent now? 
    * You can find bunch of other Gym environments here: [https://gym.openai.com/envs/](https://gym.openai.com/envs/). See if you can make these codes work with other environments (you can create environments with `gym.make([name here])`.
