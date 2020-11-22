from tensorflow import keras as k
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.python.keras.models import Model


class MockModel:

    @classmethod
    def get_model(cls) -> Model:
        # Create a fake model. Basically, we simulate a text classifier where we have 3 words which are represented with 3
        # digits: 1, 2 and 3. 0 is reserved for padding.
        # There is an embedding matrix that encode each word into a vector containing exactly one 1 representing the word itself.
        # So word 2 is represented as [0, 1, 0]
        # The classifier tells if there is the occurence of a given word. The output consists of a binary vector, where
        # the position p_i of a 1 indicates that the word i was present in the input vector.
        model = models.Sequential([
            layers.Embedding(input_dim=4, output_dim=3, input_length=3),
            layers.GlobalMaxPool1D(),
        ])

        model.compile(loss=k.losses.BinaryCrossentropy())

        model.layers[0].set_weights([np.array([[0,0,0],[1,0,0], [0, 1, 0], [0,0,1]])])

        return model

