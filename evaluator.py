"""
Created on Apr 1, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
#import formats
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
        self.input = None  # formats.Batch(device)
        self.labels = None  # formats.Labels()

    def run(self):
        a = self.input.batch
        print("(min, max, sum, avg) = (%.6f, %.6f, %.6f, %.6f)" % (a.min(), a.max(), a.sum(), numpy.average(a)))
        n_ok = 0
        n = self.input.batch.shape[0]
        for i in range(0, n):
            i_max = numpy.argmax(self.input.batch[i])
            if i_max == self.labels.batch[i]:
                n_ok += 1
        print("(n_ok, n_total): (%d, %d)" % (n_ok, n))
        if n_ok == n:
            print("Perfect")
            return 1
