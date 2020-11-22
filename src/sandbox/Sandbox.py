import tensorflow as tf
import tensorflow.keras as keras



# Investigation how the MaxPooling, GlobalMaxPooling and Conv1D are working
# ========================================================
# ========================================================

values = [[1, 2, 3, 0],[5,10,3,22]]
tf_values = tf.constant(values)
print(tf_values.get_shape())

embedder = keras.layers.Embedding(6, 2, mask_zero=True)
embedded = embedder(tf_values)

max_pool = keras.layers.MaxPooling1D()
max_pooled = max_pool(embedded)

global_max_pool = keras.layers.GlobalMaxPooling1D()
global_max_pooled = global_max_pool(embedded)

# convolution = keras.layers.Conv1D(1, 3)
# convoluted = convolution(embedded)

# print("Embedded")
# print(embedded)

print("Max pooling:")
print(max_pooled)

print("Global max pooling:")
print(global_max_pooled)

# print("Convoluted:")
# print(convoluted)
# print(convolution.get_weights())

# Difference => max pooling is some kind of convolution of the function (max()) and word with a window.


# Investigation how shuffle and batch are working together.
# ========================================================
# ========================================================


# def iterate(dataset: tf.data.Dataset, batch_count):
#     for batch in dataset:
#         print(batch.numpy())
#
#     #  for i in range(0,batch_count):
#     # #     print(list(dataset.as_numpy_iterator()))
#     #     iterator = dataset.__iter__()
#     #     try:
#     #         value = iterator.get_next()
#     #         print(value)
#     #     except tf.errors.OutOfRangeError:
#     #         pass
#
# x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#
# d = tf.data.Dataset.from_tensor_slices((x)).shuffle(15)
#
# d1_batch_size = 3
# d1_batch_count = ceil(10 / d1_batch_size)
# d2_batch_size = 4
# d2_batch_count = ceil(10 / d2_batch_size)
#
# d1 = d.take(10).shuffle(15).batch(d1_batch_size)
#
# d2 = d.skip(10).take(5).batch(d2_batch_size)
#
# print(d1)
# print(d2)
#
# print("Iterate on d1")
# iterate(d1, d1_batch_count)
#
# print("Iterate second time on d1")
# iterate(d1, d1_batch_count)
#
#
#
# print("Iterate d2")
# iterate(d2, d2_batch_count)