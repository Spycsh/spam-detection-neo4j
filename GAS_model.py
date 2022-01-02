import os
import argparse
import tensorflow as tf
from tensorflow import keras

from layers import ConcatenationAggregator, AttentionAggregator, GASConcatenation, GraphConvolution
from metrics import accuracy


class GAS(keras.Model):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass
