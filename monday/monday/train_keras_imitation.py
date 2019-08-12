#!/usr/bin/env python3
#
# train_keras_imitation.py
#
# Training keras models to do imitation learning
# 

from argparse import ArgumentParser
import random
import os
import numpy as np
import keras

# Hardcoded input shape
INPUT_SHAPE = (24, 32, 3)
# Hardcoded number of buttons
NUMBER_OF_BUTTONS = 3
# Number of training samples per update
BATCH_SIZE = 32
DEFAULT_CONFIG = "my_way_home.cfg"

class DataGenerator:
    """Generator object for loading imitation learning data"""
    def __init__(self, files, batch_size, num_to_cache=10):
        """
        files: List of paths to recordings of human play
        batch_size: Size of batches to return
        num_to_cache: Number of files to load at once
        """
        self.files = files
        self.batch_size = batch_size
        self.num_to_cache = num_to_cache
        self.x = None
        self.y = None
        self.index = 0
        self.load_data()
        
    def load_data(self):
        # Load new data from random files to memory
        files = random.sample(self.files, self.num_to_cache)
        datas = [np.load(file) for file in files]
        self.x = np.concatenate([data["states"] for data in datas])
        self.y = np.concatenate([data["actions"] for data in datas])
        # Scale [0,255] to [0,1]
        self.x = self.x.astype(np.float) / 255
        self.y = self.y.astype(np.float)
        self.index = 0

        # Make sure we have enough data for at least one batch
        if len(self.x) < self.batch_size:
            raise RuntimeError("Not enough files loaded for a batch")

    def __iter__(self):
        # Make sure there is bunch of new 
        # data to be loaded
        self.load_data()
        return self

    def __next__(self):
        if self.index + self.batch_size >= len(self.x):
            # Load new data
            self.load_data()

        x = self.x[self.index:self.index + self.batch_size]
        y = self.y[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        return (x, y)

def build_simple_keras_model(input_shape, num_buttons):
    """Creates simple keras model for predicting actions from images"""
    raise NotImplementedError("Implement Keras model below, and then remove this line")
    model = keras.models.Sequential([
        # TODO 
        # Write Keras layers here. See documentation for help: https://keras.io/ 
        # Two small convolution layers + one dense layer should do the trick.

        # Do not remove this layer (the output has to be sigmoided)
        keras.layers.Dense(num_buttons, activation="sigmoid")
    ])

    # Compile the model with losses and accuracy metrics
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def main(args):
    data_files = os.listdir(args.input_directory)
    data_files = [os.path.join(args.input_directory, data_file) 
                  for data_file in data_files]

    generator = DataGenerator(data_files, 32, num_to_cache=3)

    model = build_simple_keras_model(INPUT_SHAPE, NUMBER_OF_BUTTONS)

    # Training itself
    model.fit_generator(generator, steps_per_epoch=1000, epochs=args.epochs, use_multiprocessing=True)

    model.save(args.output)

if __name__ == "__main__":
    parser = ArgumentParser("Train Keras models to do imitation learning.")
    parser.add_argument("input_directory",
                        type=str,
                        help="Path to directory with recorded gameplay.")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs to train.")
    parser.add_argument("output",
                        type=str,
                        help="Path where to store trained model.")
    args = parser.parse_args()

    main(args)
