from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.keras.preprocessing.text import Tokenizer

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

print("####################################\n\n\n")

# good link: https://www.tensorflow.org/tutorials/keras/basic_text_classification


print(tf.version.VERSION)
print(tf.keras.__version__)

tokenizer = Tokenizer()
themeTokenizer = Tokenizer()

print("####################################\n\n\n")

texts = {
    "text":
        [
            "Samsung Galaxy S5",
            "Samsung Tab S",
            "Galaxy S",
            "Galaxy A",
            "Samsung Galaxy",
            "Galaxy note",
			"Galaxy note 2",
			"Samsung Tab 3",
			"Samsung galaxy note",
			"iphone",
            "iphone XS",
            "Apple iphone",
            "Apple iPad 3",
            "ipad 3",
            "ipad",
            "apple iphone 3gs",
            "iphone 3gs",
            "ipad air",
            "Samsung Galaxy S5",
            "Samsung Tab S",
            "Galaxy S",
            "Galaxy A",
            "Samsung Galaxy",
            "Galaxy note",
			"Galaxy note 2",
			"Samsung Tab 3",
			"Samsung galaxy note",
			"iphone",
            "iphone XS",
            "Apple iphone",
            "Apple iPad 3",
            "ipad 3",
            "ipad",
            "apple iphone 3gs",
            "iphone 3gs",
            "ipad air",
"Samsung Galaxy S5",
            "Samsung Tab S",
            "Galaxy S",
            "Galaxy A",
            "Samsung Galaxy",
            "Galaxy note",
			"Galaxy note 2",
			"Samsung Tab 3",
			"Samsung galaxy note",
			"iphone",
            "iphone XS",
            "Apple iphone",
            "Apple iPad 3",
            "ipad 3",
            "ipad",
            "apple iphone 3gs",
            "iphone 3gs",
            "ipad air",
            "Samsung Galaxy S5",
            "Samsung Tab S",
            "Galaxy S",
            "Galaxy A",
            "Samsung Galaxy",
            "Galaxy note",
			"Galaxy note 2",
			"Samsung Tab 3",
			"Samsung galaxy note",
			"iphone",
            "iphone XS",
            "Apple iphone",
            "Apple iPad 3",
            "ipad 3",
            "ipad",
            "apple iphone 3gs",
            "iphone 3gs",
            "ipad air"
        ],
    "theme":
        [
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
"samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "samsung",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
            "apple",
        ]
}

print("column theme:", texts["theme"])
print("column text:", texts["text"])

tokenizer.fit_on_texts(texts["text"])


X = tokenizer.texts_to_sequences(texts["text"])

#print("X: ", X)

themeTokenizer.fit_on_texts(texts["theme"])

Y = themeTokenizer.texts_to_sequences(texts["theme"])

# print("Y: ", Y)
#print("X back to text: ", tokenizer.sequences_to_texts(X))

voc_size = len(tokenizer.word_index) + 1
max_len_article = 5

print("Number of docs: ", tokenizer.document_count)
print("Number of words: ", voc_size)

################################################

# Padding to make all feature vector the same length.

X = keras.preprocessing.sequence.pad_sequences(X,
                                               value=0,
                                               padding='post',
                                               maxlen=max_len_article)

Y = keras.preprocessing.sequence.pad_sequences(Y,
                                               value=0,
                                               padding='post')


#print("Padded Y: ", Y)

Y = [yi for y in Y for yi in y]

Y = np.array([y-1 for y in Y])


print("Padded X: ", X)
print("Padded Y: ", Y)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset.shuffle(32)

trainSize = int(0.7 * len(X))
testSize = len(X) - trainSize

trainData = dataset.take(trainSize)
testData = dataset.skip(testSize)

firstLayoutOutputDim = 50

model = tf.keras.Sequential(
    [
        keras.layers.Embedding(input_dim=voc_size, output_dim=firstLayoutOutputDim),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(firstLayoutOutputDim, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ]
)

model.summary()

# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

print("train data: ", trainData)

# model.fit(trainData, epochs=100)
model.fit(X,Y, epochs=100)

modelEvaluationResults = model.evaluate(testData)
print("Evaluation results: ", modelEvaluationResults)


####################
# Making Predictions

def doPrediction(text):
    vector = tokenizer.texts_to_sequences([text])
    vector = keras.preprocessing.sequence.pad_sequences(vector,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=max_len_article)

    return model.predict(vector)


print(doPrediction("apple ipad air 2"))
print(doPrediction("iphone 4s"))
print(doPrediction("galaxy"))
print(doPrediction("galaxy tab"))
print("ania: ", doPrediction("That fucking ipad."))
# print(doPrediction("surface pro 2"))


print(themeTokenizer.word_index)

################################################

# if word 3 => theme 3
# if word 1 => theme 1
# if word 1 + 2 => theme 2

# a = np.array([[1, 2, 3], [1, 3, 0], [1, 2, 0], [2, 3, 0]])
# b = np.array([[1, 2, 3], [1, 3, 0], [1, 2, 0], [3, 0, 0]])
#
# dataset = tf.data.Dataset.from_tensor_slices((a, b))
# dataset.shuffle(32)
#
# trainSize = int(1 * len(a))
# testSize = len(a) - trainSize
#
# train = dataset.take(trainSize)
# test = dataset.skip(testSize)
#
#
# model2 = tf.keras.Sequential(
#     [
#         keras.layers.Embedding(10000, 16),
#         keras.layers.GlobalAveragePooling1D(),
#         keras.layers.Dense(16, activation=tf.nn.relu),
#         keras.layers.Dense(2, activation=tf.nn.sigmoid)
#     ]
# )
#
# model2.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
#
# model2.fit(train, epochs=4)
#
# # model.evaluate(test["text"], test["theme"])
#
# pred = model2.predict(np.array([[1, 3, 2]]))
#
# print("predicting: ", pred)
