import math
from typing import Any, List, Optional

import tensorflow as tf

from data_models.articles import Articles


class TrainValidationDataset:

    def __init__(self, X: List[List[Optional[Any]]], Y: List[List[Optional[Any]]], articles: Articles, validation_ratio: float, batch_size):
        """
        Creates and wrap a tensorflow dataset.
        :param X: Input
        :param Y: Outputs
        :param validation_ratio:
        :param batch_size:
        """
        if len(X) == 0:
            raise Exception("X matrix has not rows!")

        self.row_count: int = len(X)
        self.article_length: int = len(X[0])
        self.theme_count: int = len(Y[0])

        self.train_ratio: float = 1 - validation_ratio

        self.train_size = math.ceil(self.train_ratio * self.row_count)
        self.validation_size = math.ceil(validation_ratio * self.row_count)

        self.train_batch_count = int(math.ceil(self.train_size / batch_size))
        self.validation_batch_count = int(math.ceil(self.validation_size / batch_size))

        self.X_train = X[:self.train_size]
        self.Y_train = Y[:self.train_size]
        self.X_val = X[self.train_size:]
        self.Y_val = Y[self.train_size:]
        self.articles_train: Articles = Articles(articles[:self.train_size])
        self.articles_validation: Articles = Articles(articles[self.train_size:])


        # tf.Datasets creation

        # Only train shuffle. Not needed to evaluate.
        self.trainData = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train))\
            .shuffle(len(self.X_train))\
            .batch(batch_size)\
            .repeat()

        self.validationData = tf.data.Dataset.from_tensor_slices((self.X_val, self.Y_val))\
            .batch(batch_size)\
            .repeat()
