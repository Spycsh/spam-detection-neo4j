import tensorflow as tf
from tensorflow import keras

class GASModel(keras.Model):

    def __init__(self, **kwargs):
        super().__init__()

    def forward_propagation(self):
        # main algorithm here
