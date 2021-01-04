# TODO

# Finding best parameters
# =======================================================

# if DO_COMPARISONS:
#     dims = [16, 32, 64, 128, 200, 256, 512]
#     categoricalHinge = []
#     categoricalAccuracy = []
#     fig = None
#
#     for dim in dims:
#         model = createModel(dim, 128, 32)
#         modelEvaluationResults = model.evaluate(testData, steps=test_batch_count)
#
#         categoricalHinge.append(modelEvaluationResults[2])
#         categoricalAccuracy.append(modelEvaluationResults[1])
#         fig = Plot.plotLosses(dims[0:len(categoricalAccuracy)], [[categoricalAccuracy, "categoricalAccuracy"], [categoricalHinge, "categoricalHinge"]], fig)

# class ErrorData:
#
#     def __init__(self, accurary, loss, dim1, dim2, dim3, steps):
#         self.accuracy = accurary
#         self.loss = loss
#         self.dim1 = dim1
#         self.dim2 = dim2
#         self.dim3 = dim3
#         self.steps = steps
#
# if DO_COMPARISONS:
#     dims = [8, 32, 128, 256]
#     steps = [1, 5, 25, 50]
#     categoricalHinge = []
#     categoricalAccuracy = []
#     fig = None
#
#     errors = []
#     bestAccuracy = ErrorData(0, 1, -1, -1, -1, -1)
#     bestLoss = ErrorData(0, 1, -1, -1, -1, -1)
#
#     import csv
#     csvfile = open('results.csv', 'w', newline='')
#
#     fieldnames = ['accuracy', 'loss', "dim1", "dim2", "dim3", "step"]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for dim1 in dims:
#         for dim2 in dims:
#             for dim3 in dims:
#                 for step in steps:
#                     print("\n==> Doing for: %d, %d, %d, %d\n" % (dim1, dim2, dim3, step))
#
#                     model = createModel(dim1, dim2, dim3, step)
#                     modelEvaluationResults = model.evaluate(testData, steps=test_batch_count)
#                     error = ErrorData(modelEvaluationResults[1], modelEvaluationResults[2], dim1, dim2, dim3, step)
#                     errors.append(error)
#
#                     writer.writerow({'accuracy' : error.accuracy, 'loss' : error.loss, 'dim1' : dim1, 'dim2' : dim2, 'dim3' : dim3, 'step' : step})
#
#                     if error.accuracy > bestAccuracy.accuracy:
#                         bestAccuracy = error
#
#                     if error.loss < bestLoss.loss:
#                         bestLoss = error
#
#     csvfile.close()
#     print("\nDone")