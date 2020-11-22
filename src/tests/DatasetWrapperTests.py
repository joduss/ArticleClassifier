# import unittest
#
# from classifier.prediction.models.DatasetWrapper import DatasetWrapper
#
#
# class DatasetWrapperTests(unittest.TestCase):
#
#     if __name__ == '__main__':
#         unittest.main()
#
#     def testsDatasetWrapper(self):
#         X = [[1,10], [2, 20], [3, 30]]
#         Y = [[101, 110], [102, 120], [103, 130]]
#
#         dataset = DatasetWrapper(X, Y, 1/3, 1/3, 2)
#
#         # Checking the size of data.
#         self.assertEqual(3, dataset.row_count)
#         self.assertEqual(2, dataset.X_column_count)
#         self.assertEqual(2, dataset.Y_column_count)
#
#         self.assertEqual(1, dataset.train_size)
#         self.assertEqual(1, dataset.validation_size)
#         self.assertEqual(1, dataset.test_size)
#
#         self.assertEqual(1, len(dataset.X_train))
#         self.assertEqual(1, len(dataset.X_test))
#         self.assertEqual(1, len(dataset.X_val))
#
#         # Checking that the values are not used multiple times.
#         self.assertNotEqual(dataset.X_train, dataset.X_test)
#         self.assertNotEqual(dataset.X_train, dataset.X_val)
#         self.assertNotEqual(dataset.X_val, dataset.X_test)
#
#         # Checking that shuffling data shuffles X and Y in pair!
#         self.assertEqual([x + 100 for x in dataset.X_test[0]], dataset.Y_test[0])
#         self.assertEqual([x + 100 for x in dataset.X_train[0]], dataset.Y_train[0])
#         self.assertEqual([x + 100 for x in dataset.X_val[0]], dataset.Y_val[0])
#
#         # Checking the content of the datasets
#         test_data_list = list(dataset.testData.take(1).as_numpy_iterator())[0]
#         self.assertEqual(dataset.X_test[0], test_data_list[0][0].tolist())
#         self.assertEqual(dataset.Y_test[0], test_data_list[1][0].tolist())
#
#         val_data_list = list(dataset.validationData.take(1).as_numpy_iterator())[0]
#         self.assertEqual(dataset.X_val[0], val_data_list[0][0].tolist())
#         self.assertEqual(dataset.Y_val[0], val_data_list[1][0].tolist())
#
#         train_data_list = list(dataset.trainData.take(1).as_numpy_iterator())[0]
#         self.assertEqual(dataset.X_train[0], train_data_list[0][0].tolist())
#         self.assertEqual(dataset.Y_train[0], train_data_list[1][0].tolist())
#
