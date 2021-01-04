import math

import tensorflow as tf

class GlobalDataset:

    def __init__(self, tf_dataset: tf.data.Dataset, train_ratio: float, validation_ratio: float, batch_size: int = 300):

        self.article_length = len(list(tf_dataset.as_numpy_iterator())[0][0])
        self.theme_count = len(list(tf_dataset.as_numpy_iterator())[0][1])
        self.count = len(list(tf_dataset.as_numpy_iterator()))

        self.dataset = tf_dataset.batch(batch_size).repeat().shuffle(batch_size)


        self.trainSize = int(train_ratio * self.count)
        self.validationSize = int(validation_ratio * self.count)
        self.testSize = self.count - self.trainSize - self.validationSize

        self.trainData = self.dataset.take(self.trainSize).repeat()
        self.validationData = self.dataset.skip(self.trainSize).take(self.validationSize).repeat()
        self.testData = self.dataset.skip(self.testSize)

        self.train_batch_count = int(math.ceil(self.trainSize / batch_size))
        self.test_batch_count = int(math.ceil(self.testSize / batch_size))
        self.validation_batch_count = int(math.ceil(self.validationSize / batch_size))


    # def __init__(self, X, Y, train_ratio: float, validation_ratio: float, batch_size):
    #     """
    #     Creates and wrap a tensorflow dataset.
    #     :param X: Input
    #     :param Y: Outputs
    #     :param train_ratio:
    #     :param validation_ratio:
    #     :param batch_size:
    #     """
    #     self.X_column_count = len(X[0])
    #     self.Y_column_count = len(Y[0])
    #     self.row_count = len(X)
    #
    #     XY = list(zip(X, Y))
    #     shuffle(XY)
    #     X, Y = zip(*XY)
    #
    #     self.train_size = math.ceil(train_ratio * self.row_count)
    #     self.validation_size = math.ceil(validation_ratio * self.row_count)
    #     self.test_size = self.row_count - self.train_size - self.validation_size
    #
    #     self.train_batch_count = int(math.ceil(self.train_size / batch_size))
    #     self.test_batch_count = int(math.ceil(self.test_size / batch_size))
    #     self.validation_batch_count = int(math.ceil(self.validation_size / batch_size))
    #
    #     self.X_test = X[0:self.test_size]
    #     self.Y_test = Y[0:self.test_size]
    #
    #     self.X_train = X[self.test_size:self.test_size + self.train_size]
    #     self.Y_train = Y[self.test_size:self.test_size + self.train_size]
    #     self.X_val = X[self.test_size + self.train_size:]
    #     self.Y_val = Y[self.test_size + self.train_size:]
    #
    #     # tf.Datasets creation
    #
    #     # Only train shuffle. Not needed to evaluate/test
    #     self.trainData = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train)).batch(batch_size).repeat()
    #     self.validationData = tf.data.Dataset.from_tensor_slices((self.X_val, self.Y_val)).batch(batch_size).repeat()
    #     self.testData = tf.data.Dataset.from_tensor_slices((self.X_test, self.Y_test)).batch(batch_size).repeat()
