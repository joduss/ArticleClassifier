from dataclasses import dataclass
from typing import List

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.metrics as metrics
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.models import Model

from classifier.Data.TrainValidationDataset import TrainValidationDataset
from classifier.prediction.losses.weightedBinaryCrossEntropy import WeightedBinaryCrossEntropy
from classifier.models.IClassifierModel import IClassifierModel
from classifier.models.utility.ManualInterrupter import ManualInterrupter


@dataclass
class ClassifierModel3(IClassifierModel):

    # Will contain the model once trained.
    __model__: Model

    # Model properties
    __model_name__ = "Model-3"
    run_eagerly: bool = False


    def __init__(self):
        pass


    def train_model(self, themes_weight: List[float], dataset: TrainValidationDataset, voc_size: int, keras_callback: LambdaCallback):
        epochs = 60
        embedding_output_dim = 128
        last_dim = 128

        article_length = dataset.article_length
        theme_count = dataset.theme_count

        model = tf.keras.Sequential(
            [
                keras.layers.Embedding(input_dim=voc_size, input_length=article_length, output_dim=embedding_output_dim,
                                       mask_zero=True),
                keras.layers.Conv1D(filters=64, kernel_size=3, input_shape=(voc_size, embedding_output_dim),
                                    activation=tf.nn.relu),
                keras.layers.GlobalMaxPooling1D(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(last_dim, activation=tf.nn.relu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(theme_count, activation=tf.nn.sigmoid, kernel_regularizer=regularizers.l2(0.2),
                                   activity_regularizer=regularizers.l1(0.1))
            ]
        )

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1, clipvalue=0.5),
                      loss=WeightedBinaryCrossEntropy(themes_weight, from_logits=True),
                      metrics=[metrics.AUC(), metrics.BinaryAccuracy(), metrics.TruePositives(),
                               metrics.TrueNegatives(), metrics.FalseNegatives(), metrics.FalsePositives(),
                               metrics.Recall(), metrics.Precision()],
                      run_eagerly=self.run_eagerly)

        keras.utils.plot_model(model, "output/" + self.model_name + ".png", show_shapes=True)

        model.fit(dataset.trainData, epochs=epochs, steps_per_epoch=dataset.train_batch_count,
                  validation_data=dataset.validationData, validation_steps=dataset.validation_batch_count,
                  callbacks=[ManualInterrupter(), keras_callback])

        model.save("output/" + self.model_name + ".h5")
        model.save_weights("output/" + self.model_name + "_weight.h5")

        self.__model__ = model


    def get_model_name(self) -> str:
        return self.__model_name__

    def get_keras_model(self) -> Model:
        if self.__model__ is None:
            raise Exception("The model must first be trained!")
        return self.__model__

    def stop(self, model):
        model.stop_training = True
