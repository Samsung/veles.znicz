"""
Created on Apr 1, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters


class BatchEvaluator(filters.OpenCLFilter):
    """Evaluator for nn softmax output from the batch labels.

    Attributes:
        labels: labels for batch.
    """
    def __init__(self, device=None, unpickling = 0):
        super(BatchEvaluator, self).__init__(unpickling=unpickling, device=device)
        if unpickling:
            return
        self.labels = filters.Labels()

    def run(self):
        print("TODO(a.kazantsev): implement " + self.__class__.__name__ + "::run()")
