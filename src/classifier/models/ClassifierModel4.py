import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.metrics import AUC, BinaryAccuracy, FalseNegatives, FalsePositives, Precision, Recall, \
    TrueNegatives, TruePositives
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.layers import Dropout

from classifier.Data.TrainValidationDataset import TrainValidationDataset
from classifier.models.IClassifierModel import IClassifierModel
from classifier.models.utility.ManualInterrupter import ManualInterrupter
from classifier.prediction.losses.weightedBinaryCrossEntropy import WeightedBinaryCrossEntropy
from data_models.weights.theme_weights import ThemeWeights


class ClassifierModel4(IClassifierModel):

    embedding_output_dim = 128
    epochs = 15

    # Will contain the model once trained.
    __model__: Model

    # Model properties
    __model_name__ = "Model-4"
    run_eagerly: bool = False

    # Other properties
    __plot_directory: str

    def __init__(self, plot_directory: str = None):
        self.__plot_directory = plot_directory


    def train_model(self, themes_weight: ThemeWeights, dataset: TrainValidationDataset, voc_size: int, keras_callback: LambdaCallback):

        article_length = dataset.article_length
        theme_count = dataset.theme_count

        model = tf.keras.Sequential(
            [
                keras.layers.Embedding(input_dim=voc_size, input_length=article_length, output_dim=self.embedding_output_dim,
                                       mask_zero=True),
                Dropout(0.3),
                keras.layers.Conv1D(filters=64, kernel_size=3, input_shape=(voc_size, self.embedding_output_dim),
                                    activation=tf.nn.relu),
                #keras.layers.MaxPooling1D(3),
                #keras.layers.Bidirectional(keras.layers.LSTM(64)),
                keras.layers.GlobalAveragePooling1D(),
                Dropout(0.3),
                keras.layers.Dense(theme_count, activation=tf.nn.sigmoid)
            ]
        )


        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1),
                      loss=WeightedBinaryCrossEntropy(themes_weight.weight_array()),
                      metrics=[AUC(multi_label=True), BinaryAccuracy(), TruePositives(),
                         TrueNegatives(), FalseNegatives(), FalsePositives(),
                         Recall(), Precision()],
                      run_eagerly=True)

        model.summary()
        self.__model__ = model

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
                  callbacks=callbacks)


    def get_model_name(self):
        return self.__model_name__

    def get_keras_model(self) -> Model:
        if self.__model__ is None:
            raise Exception("The model must first be trained!")
        return self.__model__