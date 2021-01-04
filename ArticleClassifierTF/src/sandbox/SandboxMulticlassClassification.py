from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.keras.preprocessing.text import Tokenizer

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import math

print("####################################\n\n\n")

# good link: https://www.tensorflow.org/tutorials/keras/basic_text_classification


print(tf.version.VERSION)
print(tf.keras.__version__)

t = Tokenizer()
txts = ["apple", "micro"]

t.fit_on_texts(txts)
mt = t.texts_to_matrix(txts)
print(t.texts_to_matrix(txts))

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
            "ipad air",
            "surface",
            "microsoft surface",
            "Surface pro",
            "microsoft surface light",
            "surface 2",
            "surface book",
            "Microsoft surface book 2",
            "microsoft surface pro",
            "surface touch",
            "surface 3"
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
            "microsoft",
            "microsoft",
            "microsoft",
            "microsoft",
            "microsoft",
            "microsoft",
            "microsoft",
            "microsoft",
            "microsoft",
            "microsoft"
        ]
}

print("column theme:", texts["theme"])
print("column text:", texts["text"])

tokenizer.fit_on_texts(texts["text"])


X = tokenizer.texts_to_sequences(texts["text"])

#print("X: ", X)

themeTokenizer.fit_on_texts(texts["theme"])

Y = themeTokenizer.texts_to_matrix(texts["theme"])

# print("Y: ", Y)
#print("X back to text: ", tokenizer.sequences_to_texts(X))

voc_size = len(tokenizer.word_index) + 1
theme_count = len(themeTokenizer.word_index) + 1
max_len_article = 7

print("Number of docs: ", tokenizer.document_count)
print("Number of words: ", voc_size)

################################################

# Padding to make all feature vector the same length.

X = keras.preprocessing.sequence.pad_sequences(X,
                                               value=0,
                                               padding='post',
                                               maxlen=max_len_article)

# Y = keras.preprocessing.sequence.pad_sequences(Y,
#                                                value=0,
#                                                padding='post')

#print("Padded X: ", X)
#print("Padded Y: ", Y)

# Create dataset

dataset = tf.data.Dataset.from_tensor_slices((X,Y))

BATCH_SIZE = 5

dataset = tf.data.Dataset.from_tensor_slices((X,Y))
dataset = dataset.batch(BATCH_SIZE).repeat().shuffle(BATCH_SIZE)

trainSize = int(0.7 * len(X))
testSize = len(X) - trainSize

trainData = dataset.take(trainSize).repeat()
testData = dataset.skip(testSize)

train_batch_count = int(math.ceil(trainSize / BATCH_SIZE))
test_batch_count = int(math.ceil(testSize / BATCH_SIZE))

firstLayoutOutputDim = 50

model = tf.keras.Sequential(
    [
        keras.layers.Embedding(input_dim=voc_size, output_dim=firstLayoutOutputDim),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(firstLayoutOutputDim, activation=tf.nn.relu),
        # keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        keras.layers.Dense(theme_count, activation=tf.nn.softmax)
    ]
)

model.summary()

# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

print("train data: ", trainData)

model.fit(trainData, epochs=100, steps_per_epoch=train_batch_count)
#model.fit(X,Y, epochs=100)


print("\nPerform evaluation---")
modelEvaluationResults = model.evaluate(testData, steps=test_batch_count)
#modelEvaluationResults = model.evaluate(X,Y)

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


print("apple ipad air 2", doPrediction("apple ipad air 2"))
print("iphone 4s", doPrediction("iphone 4s"))
print("galaxy", doPrediction("galaxy"))
print("galaxy tab", doPrediction("galaxy tab"))
print("ania: \That fucking ipad.\"", doPrediction("That fucking ipad."))
print("surface special: ", doPrediction("surface special"))
print("surface go: ", doPrediction("surface go"))
print("microsoft et apple: ", doPrediction("microsoft et apple."))
print("Je suis sur la lune: ", doPrediction("Je suis sur la lune"))

print(themeTokenizer.word_index)

