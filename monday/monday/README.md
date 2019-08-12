# Monday practicals

Topic: Imitation learning (behavioral cloning) in Doom.

## Files

* record_vizdoom.py: Record human gameplay in ViZDoom scenarios and store on disk (images, actions and reward).
* train_keras_imitation.py: Create neural networks with Keras and train on human data gathered with previous script.
* enjoy_keras_imitation.py: Enjoy/Evaluate trained models with previous script.
* enjoy_random.py: Enjoy/Evaluate random agent

## Tasks

All coding tasks are marked with `# TODO`s in the code.

Playing game is done with arrow keys.

1. Implement missing code in `record_vizdoom.py` and record data of yourself playing with `record_vizdoom.py`. Ten games should already work decently.
2. Implement a simple Keras network in `train_keras_imitation.py`, e.g. two convolutional layers followed by a dense layer. 
   See Keras documentation for help: https://keras.io/
3. Train network to play like you with `train_keras_imitation.py`.
4. See how well your agent does with `enjoy_keras_imitation.py`. You can also run code with `--evaluate` to get more objective results.
5. Implement random actions in `enjoy_random.py` (each step, each button is randomly pressed (1) or not (0)).
6. Run `enjoy_random.py` and compare performance of your trained model to random agent.
7. Improve the performance of your agent. Can you reach 70% success rate? Things to think about:  
    * Is your training data good? Remember, this model will follow your steps **exactly**.
    * Did you overfit your model? You can detect this by having separate testing set (not trained on), or from too high accuracy during training.
        * If so, how to fight it? Core idea is to prevent model from memorizing each training sample.

If you have time, you can try same code on different ViZDoom scenarios. You can find bunch of them here: 
https://github.com/mwydmuch/ViZDoom/tree/master/scenarios
