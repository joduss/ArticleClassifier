import tensorflow as tf
import tensorflow.keras as keras

# rule: out 1: si in1 >= 10 => 1
# rule out2: si in2 >= 20 => 1
from tensorflow.python.keras.losses import BinaryCrossentropy

from classifier.prediction.losses.weightedBinaryCrossEntropy import WeightedBinaryCrossEntropy

# data_in = [[0, 6], [11, 6], [1, 1], [9, 0], [20, 20], [20, 22], [0, 19], [5, 17], [10, 9], [18, 3],
#            [11, 19], [11, 21], [9,19], [9,21], [15,15], [18,22], [22,18], [3,3], [8,18], [0, 0]]
# data_out = [[0, 0], [1, 0], [0, 0], [0, 0], [1, 1],   [1, 1],   [0, 0],   [0, 0], [1, 0],  [1, 0],
#             [1, 0], [1,1], [0,0], [0,1], [1,0], [1,1], [1,0], [0,0], [0,0], [0,0]]

data_in = [[0, 6], [11, 6], [1, 1], [9, 0], [20, 20], [20, 22], [0, 19], [5, 17], [10, 9], [18, 3],[11, 19], [11, 21], [9,19],  [0,21], [15,15], [18,22], [22,18], [3,3], [8,18], [0, 0],[0, 6], [11, 6], [9, 10], [9, 19], [10, 20], [10, 22], [4, 17], [5, 17], [10, 16], [18, 3],[11, 19], [12, 21], [8,0], [2,20],[10,11], [14, 20], [22,0], [5,18], [2,2], [4, 19]]
data_out = [[0, 0], [1, 0], [0, 0], [0, 0], [1, 1],   [1, 1],   [0, 0],  [0, 0],  [1, 0],  [1, 0], [1, 0],    [1,1],   [0,0],   [0,1], [1,0],    [1,1],  [1,0],   [0,0], [0,0],  [0,0], [0, 0], [1, 0],  [0, 0],  [0, 0],  [1, 1],   [1, 1],   [0, 0],  [0, 0],  [1, 0],   [1, 0], [1, 0],   [1,1],    [0,0], [0,1], [1,0],   [1,1],    [1,0],  [0,0],  [0,0], [0,0]]

in_tf = tf.constant(data_in)
out_tf = tf.constant(data_out)
out_tf = tf.reshape(out_tf, [40,2,1])

# class 1: 10 / 20
# class 2: 5 / 20

model = keras.models.Sequential([
    keras.layers.Dense(2, activation=keras.activations.relu, kernel_initializer=keras.initializers.Ones),
    keras.layers.Dense(2, activation=keras.activations.sigmoid, kernel_initializer=keras.initializers.Ones)
])

# input = keras.layers.Input(shape=(2), name="Input")
#
# dense = keras.layers.Dense(2, activation=keras.activations.relu)(input)
# out1 = keras.layers.Dense(1, activation=keras.activations.sigmoid, name="out1")(dense)
# out2 = keras.layers.Dense(1, activation=keras.activations.sigmoid, name="out2")(dense)

# model = keras.Model(inputs=[input], outputs=[out1, out2])

model.build(input_shape=in_tf.shape)
model.summary()

# model.compile(loss=BinaryCrossentropy(), metrics=keras.metrics.BinaryAccuracy(), run_eagerly=True)

model.compile(loss=WeightedBinaryCrossEntropy([[1,1],[0.66666,2]]), metrics=keras.metrics.BinaryAccuracy(), run_eagerly=False, optimizer=keras.optimizers.Adam())
model.fit(x=data_in, y=data_out, epochs=2500)
# model.fit(x=data_in, y=data_out, epochs=1000, class_weight={"out1" : {0: 1, 1: 1}, "out2" : {0 : 0.666, 1: 2}})


print(f"[11,27] => {model.predict([[12,27]])}")
print(f"[0,0] => {model.predict([[0,0]])}")
print(f"[6,18] => {model.predict([[6,22]])}")
print(f"[9,22] => {model.predict([[9,22]])}")
print(f"[9,18] => {model.predict([[9,18]])}")
print(f"[10,20] => {model.predict([[10,20]])}")
print(f"[0,22] => {model.predict([[0,22]])}")
print(f"[0,26] => {model.predict([[0,26]])}")
print(f"[0,18] => {model.predict([[0,18]])}")
print(f"[8,0] => {model.predict([[8,0]])}")
print(f"[12,0] => {model.predict([[12,0]])}")
print(f"[6,0] => {model.predict([[6,0]])}")
print(f"[3,0] => {model.predict([[3,0]])}")
