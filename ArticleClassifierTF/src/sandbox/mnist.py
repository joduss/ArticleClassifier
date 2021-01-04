import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
one_hot_y_train = tf.one_hot(np.array([0,1,1,2]).astype(np.int32), depth=10)

print(y_train)
print(one_hot_y_train)