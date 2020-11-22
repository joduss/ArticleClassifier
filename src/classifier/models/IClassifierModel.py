from tensorflow import keras
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.models import Model

from classifier.Data.TrainValidationDataset import TrainValidationDataset
from data_models.weights.theme_weights import ThemeWeights


class IClassifierModel:

    _model: keras.Model

    def train_model(self, themes_weight: ThemeWeights, dataset: TrainValidationDataset, voc_size: int, keras_callback: LambdaCallback):
        """
        Creates and train the model.
        :param themes_weight:
        :param dataset:
        :param voc_size:
        :return:
        """
        raise Exception("Not implemented.")


    def get_model_name(self) -> str:
        """
        Return the name of the model.
        """
        raise Exception("Not implemented.")

    def get_keras_model(self) -> Model:
        if self._model is None:
            raise Exception("The model must first be trained!")
        return self._model

    def save_model(self, directory: str):
        self._model.save(directory + self.get_model_name() + ".h5")
        #self.get_keras_model().save_weights(directory + self.get_model_name() + "_weight.h5")

    def load_model(self, directory: str):
        self._model = keras.models.load_model(directory + self.get_model_name() + ".h5")
        # model.load_weights(directory + self.get_model_name() + "_weight.h5")

    def plot_model(self, directory: str):
        """
        Plot the model and save the image in the given directory.
        """
        keras.utils.plot_model(self.get_keras_model(),
                               directory + self.get_model_name() + '.png',
                               show_shapes=True)
