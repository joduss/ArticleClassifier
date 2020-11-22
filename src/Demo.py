import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# X = [[1, 3], [10, 15], [4,2], [6, 3], [0,0], [0,1], [1, 0]]
# Y = [  4,       26,       6,     10,     0,     1,    1]
#
# model = keras.Sequential([
#   layers.Dense(1, input_dim=2)
# ])
#
# model.summary()
#
#
# keras.utils.plot_model(model, 'demo.png', show_shapes=True)
#
#
# model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.RMSprop())
# model.fit(x=X, y=Y, epochs=2000)
#
# predictions = model.predict([[1,2], [10,0],  [0,10]])
#
# print(predictions)



# Classification

X = [[1, 3], [10, 15], [4,2], [20, 4], [0,5], [0,1], [1, 0], [51, 0], [49, 0], [0, 4.9]]
Y = [  0,         1,     0,     1 ,      1,     0,      0,      1,       0,      0]

model = keras.Sequential([
  layers.Dense(1, input_dim=2, activation=keras.activations.sigmoid)
])

model.summary()


keras.utils.plot_model(model, 'demo.png', show_shapes=True)


model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam())
model.fit(x=X, y=Y, epochs=2000)

predictions = model.predict([[12,4], [6,0],  [0,6], [1, 2], [7, 0], [30, 10], [0,0]])

print(predictions)