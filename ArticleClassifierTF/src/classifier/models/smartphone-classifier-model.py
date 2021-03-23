import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.metrics import AUC, BinaryAccuracy, FalseNegatives, FalsePositives, Precision, Recall, \
    TrueNegatives, TruePositives
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.regularizers import l2

from classifier.Data.TrainValidationDataset import TrainValidationDataset
from classifier.models.IClassifierModel import IClassifierModel
from classifier.models.utility.ManualInterrupter import ManualInterrupter
from classifier.prediction.losses.weighted_binary_cross_entropy import WeightedBinaryCrossEntropy
from data_models.weights.theme_weights import ThemeWeights

class ClassifierModel5(IClassifierModel):

    embedding_output_dim = 50
    epochs = 250

    # Model properties
    __model_name__ = "Model-5"
    run_eagerly: bool = False

    # Other properties
    __plot_directory: str

    def __init__(self, plot_directory: str = None):
        self.__plot_directory = plot_directory

# Evaluation of predictions for theme "smartphone"
# ------
# TP = 380
# TN = 551
# FP = 66
# FN = 64
# Recall = 0.8558558558558559
# Precision = 0.852017937219731
# F-scrore = 0.853932584269663
# AUC = 0.9380064829821717
#
class SmartphoneClassifierModel(IClassifierModel):

    embedding_output_dim = 50
    epochs = 60

    # Model properties
    __model_name__ = "Model-Smartphone"
    run_eagerly: bool = False

    # Other properties
    __plot_directory: str

    def __init__(self, plot_directory: str = None):
        self.__plot_directory = plot_directory


    def train_model(self, themes_weight: ThemeWeights,
                    dataset: TrainValidationDataset,
                    voc_size: int,
                    keras_callback: LambdaCallback):

        conv_reg = 0#0.015
        dense_reg = 0#0.015
        dropout_conv = 0.1#0.2
        dropout_dense = 0.1#0.2

        article_length = dataset.article_length
        theme_count = dataset.theme_count

        input = keras.layers.Input(shape=(dataset.article_length,))

        layer = keras.layers.Embedding(input_dim=voc_size, input_length=article_length, output_dim=self.embedding_output_dim,
                                       mask_zero=True)(input)
        layer = Dropout(dropout_conv)(layer)

        # Each convolution will have filter looking for some specific word combinations. For example a filter might
        # return a small value except for "apple iphone".
        conv1 = keras.layers.Conv1D(filters=32, kernel_size=2, input_shape=(voc_size, self.embedding_output_dim),
                                    activation=tf.nn.relu, kernel_regularizer=l2(l=conv_reg))(layer)
        conv1 = keras.layers.GlobalMaxPooling1D()(conv1)
        conv1 = keras.layers.Dense(16, activation=tf.nn.relu)(conv1)
        conv1 = Dropout(dropout_conv)(conv1)

        conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, input_shape=(voc_size, self.embedding_output_dim),
                                    activation=tf.nn.relu, kernel_regularizer=l2(l=conv_reg))(layer)
        conv2 = keras.layers.GlobalMaxPooling1D()(conv2)
        conv2 = keras.layers.Dense(16, activation=tf.nn.relu)(conv2)
        conv2 = Dropout(dropout_conv)(conv2)

        conv3 = keras.layers.Conv1D(filters=32, kernel_size=1, input_shape=(voc_size, self.embedding_output_dim),
                                    activation=tf.nn.relu, kernel_regularizer=l2(l=conv_reg))(layer)
        conv3 = keras.layers.GlobalMaxPooling1D()(conv3)
        conv3 = keras.layers.Dense(16, activation=tf.nn.relu)(conv3)
        conv3 = Dropout(dropout_conv)(conv3)

        conv4 = keras.layers.Conv1D(filters=32, kernel_size=5, input_shape=(voc_size, self.embedding_output_dim),
                                    activation=tf.nn.relu, kernel_regularizer=l2(l=conv_reg))(layer)
        conv4 = keras.layers.GlobalMaxPooling1D()(conv4)
        conv4 = keras.layers.Dense(16, activation=tf.nn.relu)(conv4)
        conv4 = Dropout(dropout_conv)(conv4)

        layer = keras.layers.Concatenate()([conv1, conv2, conv3, conv4])
        layer = keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=l2(l=conv_reg))(layer)
        layer = keras.layers.Dropout(dropout_dense)(layer)
        # layer = keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=l2(l=conv_reg))(layer)
        # layer = keras.layers.Dropout(dropout_dense)(layer)

        layer = keras.layers.Dense(theme_count, activation=tf.nn.sigmoid, kernel_regularizer=l2(l=dense_reg))(layer)

        model = keras.Model(inputs=input, outputs=layer)


        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=0.00009),
                      loss=WeightedBinaryCrossEntropy(themes_weight.weight_array()),
                      metrics=[AUC(multi_label=True), BinaryAccuracy(), TruePositives(),
                         TrueNegatives(), FalseNegatives(), FalsePositives(),
                         Recall(), Precision()],
                      run_eagerly=None)

        model.summary()
        self._model = model

        if self.__plot_directory is not None:
            self.plot_model(self.__plot_directory)

        # Fix for https://github.com/tensorflow/tensorflow/issues/38988
        model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

        callbacks = [ManualInterrupter(), keras_callback]

        model.fit(dataset.trainData,
                  epochs=self.epochs,
                  steps_per_epoch=dataset.train_batch_count,
                  validation_data=dataset.validationData,
                  validation_steps=dataset.validation_batch_count,
                  callbacks=callbacks,
                  )


    def get_model_name(self):
        return self.__model_name__