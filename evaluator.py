"""
Created on Apr 1, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import numpy


class BatchEvaluator(filters.OpenCLFilter):
    """Evaluator for nn softmax output from the batch labels.

    Attributes:
        input: input as Batch.
        labels: labels for Batch.
    """
    def __init__(self, device=None, unpickling = 0):
        super(BatchEvaluator, self).__init__(unpickling=unpickling, device=device)
        if unpickling:
            return
        self.input = filters.Batch()
        self.labels = filters.Labels()

    def run(self):
        print("(min, max, sum, average) = (%.6f, %.6f, %.6f, %.6f)" % \
              (self.input.batch.min(), self.input.batch.max(), self.input.batch.sum(), \
               numpy.average(self.input.batch)))
        print("TODO(a.kazantsev): implement " + self.__class__.__name__ + "::run()")
