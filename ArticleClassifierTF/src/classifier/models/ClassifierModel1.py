from dataclasses import dataclass
from typing import List

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.metrics as metrics
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.models import Model

from classifier.Data.TrainValidationDataset import TrainValidationDataset
from classifier.models.IClassifierModel import IClassifierModel
from classifier.models.utility.ManualInterrupter import ManualInterrupter


@dataclass
class ClassifierModel1(IClassifierModel):

    # Will contain the model once trained.
    __model__: Model

    # Model properties
    __model_name__ = "Model-1"
    run_eagerly: bool = False


    def __init__(self):
        pass


    def train_model(self, themes_weight: List[float], dataset: TrainValidationDataset, voc_size: int, keras_callback: LambdaCallback):

        article_length = dataset.article_length
        theme_count = dataset.theme_count

        model = tf.keras.Sequential(
            [
                # 1
                # keras.layers.Embedding(input_dim=voc_size, output_dim=firstLayoutOutputDim),
                # keras.layers.Dropout(0.2),
                # keras.layers.Conv1D(200,3,input_shape=(ARTICLE_MAX_WORD_COUNT,firstLayoutOutputDim), activation=tf.nn.relu),
                # keras.layers.GlobalAveragePooling1D(),
                # keras.layers.Dense(250, activation=tf.nn.relu),
                # keras.layers.Dense(theme_count, activation=tf.nn.softmax)

                # 2
                # keras.layers.Embedding(input_dim=voc_size, output_dim=firstLayoutOutputDim),
                # keras.layers.LSTM(ltsmOutputDim, dropout=0.2, recurrent_dropout=0.2, activation='tanh'),
                # keras.layers.Dense(theme_count, activation=tf.nn.softmax)

                # 3
                # keras.layers.Embedding(input_dim=self.voc_size, output_dim=embedding_output_dim),
                # keras.layers.Bidirectional(keras.layers.LSTM(intermediate_dim, return_sequences=True)),
                # # keras.layers.Dropout(0.1),
                # keras.layers.Bidirectional(keras.layers.LSTM(last_dim, dropout=0.05, recurrent_dropout=0.05)),
                # keras.layers.Dense(last_dim, activation=tf.nn.relu),
                # keras.layers.Dense(self.theme_count, activation=tf.nn.softmax)

                # 4
                # keras.layers.Embedding(input_dim=self.voc_size, input_length=self.article_length, output_dim=embedding_output_dim),
                # keras.layers.Bidirectional(keras.layers.LSTM(intermediate_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
                # keras.layers.Dropout(0.2),
                # keras.layers.Bidirectional(keras.layers.LSTM(last_dim * 2, recurrent_dropout=0.2)), #was last_dim * 2
                # keras.layers.Dense(last_dim, activation=tf.nn.relu),
                # keras.layers.Dense(self.theme_count, activation=tf.nn.sigmoid)

                # 5
                #keras.layers.Embedding(input_dim=self.voc_size, input_length=self.article_length, output_dim=embedding_output_dim),
                # keras.layers.Conv1D(filters=64, kernel_size=5, input_shape=(self.voc_size, embedding_output_dim), activation="relu"),
                # keras.layers.MaxPool1D(4),
                #keras.layers.Bidirectional(keras.layers.LSTM(intermediate_dim, recurrent_dropout=0.1)),
                #keras.layers.Dense(last_dim, activation=tf.nn.relu),
                #keras.layers.Dense(self.theme_count, activation=tf.nn.sigmoid)

                #6
                keras.layers.Embedding(input_dim=voc_size, input_length=article_length, output_dim=128, mask_zero=True),
                keras.layers.Bidirectional(keras.layers.LSTM(128, recurrent_dropout=0.2, dropout=0.2)),
                #keras.layers.Dropout(0.2),
                #keras.layers.Dense(last_dim, activation=tf.nn.relu),
                # keras.layers.Dense(self.theme_count, activation=tf.nn.sigmoid, use_bias=True,bias_initializer=tf.keras.initializers.Constant(-1.22818328))
                keras.layers.Dense(theme_count, activation=tf.nn.sigmoid, kernel_regularizer=regularizers.l2(0.1),
                                        activity_regularizer=regularizers.l1(0.05))

                # 7
                # keras.layers.Embedding(input_dim=self.voc_size, input_length=self.article_length,
                #                        output_dim=embedding_output_dim),
                # keras.layers.GlobalAvgPool1D(),
                # keras.layers.Dense(last_dim, activation=tf.nn.relu),
                # keras.layers.Dense(self.theme_count, activation=tf.nn.sigmoid)
            ]
        )

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1, clipvalue=0.5),
                      #loss=WeightedBinaryCrossEntropy(themes_weight, from_logits=True),
                      loss=keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[metrics.AUC(), metrics.BinaryAccuracy(), metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalseNegatives() , metrics.FalsePositives(), metrics.Recall(), metrics.Precision()],
                      run_eagerly=self.run_eagerly)

        keras.utils.plot_model(model, 'Model1.png', show_shapes=True)

        cb_list = [ManualInterrupter, keras_callback]

        model.fit(dataset.trainData, epochs=10, steps_per_epoch=dataset.train_batch_count,
                  validation_data=dataset.validationData, validation_steps=dataset.validation_batch_count,
                  callbacks=cb_list, class_weight={0:1, 1:themes_weight[0]})

        model.save("output/" + self.get_model_name() + ".h5")
        model.save_weights("output/" + self.get_model_name() + "_weight.h5")

        self.__model__ = model


    def get_model_name(self) -> str:
        return self.__model_name__


    def get_keras_model(self) -> Model:
        if self.__model__ is None:
            raise Exception("The model must first be trained!")
        return self.__model__


    def stop(self, model):
        model.stop_training = True



