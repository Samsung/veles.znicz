"""
Created on Apr 1, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters


class BatchEvaluator(filters.OpenCLFilter):
    """Evaluator for nn softmax output from the batch labels.
    """
    def __init__(self, device=None, unpickling = 0):
        super(BatchEvaluator, self).__init__(unpickling=unpickling, device=device)

    def feed_from_batch(self, src):
        """Evaluate src softmax output from the batch labels.
        """
        labels = src.output.labels
        print(labels.n_classes)
        print("TODO(a.kazantsev): "+self.__class__.__name__+"::feed_from_batch()")

        self.update_mtime()
        if self.parent:
            self.parent.child_changed(self)

    def input_changed(self, src):
        """GeneralFilter method.
        """
        if src.output.__class__.__name__ == "DataBatch":
            return self.feed_from_batch(src)
