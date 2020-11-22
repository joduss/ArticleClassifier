from typing import List

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.losses import BinaryCrossentropy


class WeightedBinaryCrossEntropy(BinaryCrossentropy):
    """
    Implementation follows https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
    """


    def __init__(self, weights: List[List[float]]):
        super().__init__()
        self.weights = tf.constant(weights)


    def call(self, y_true, y_pred):
        return \
            K.mean(
                (self.weights[:, 0] ** (1 - y_true)) * (self.weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true,y_pred),
                axis=-1
            )
