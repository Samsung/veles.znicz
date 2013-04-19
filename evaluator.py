"""
Created on Apr 1, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import formats
import numpy
import time


class BatchEvaluator(filters.OpenCLFilter):
    """Evaluator for nn softmax output from the batch labels.

    Attributes:
        labels: labels for Batch.
        y: output of the network as Batch.
        err_y: backpropagation errors based on labels.
        status: status of the evaluation (status.completed = True when learning ended).
        threshold: threshold for skipping trained well enough samples.
    """
    def __init__(self, threshold = 0.5, device = None, unpickling = 0):
        super(BatchEvaluator, self).__init__(unpickling=unpickling, device=device)
        if unpickling:
            return
        self.labels = None  # formats.Labels()
        self.y = None  # formats.Batch(device)
        self.err_y = formats.Batch()
        self.status = filters.Connector()
        self.status.completed = False
        self.threshold = threshold

    def initialize(self):
        if self.err_y.batch == None or self.err_y.batch.size != self.y.batch.size:
            self.err_y.batch = filters.aligned_zeros(self.y.batch.shape)
            self.err_y.batch_ = None

        if not self.device:
            return

        self.err_y.initialize(self.device)

    def cpu_run(self):
        t1 = time.time()

        self.y.sync()
        n_ok = 0
        batch_size = self.y.batch.shape[0]
        labels = self.labels.batch
        for i in range(0, batch_size):  # loop by batch
            y = self.y.batch[i]
            y = y.reshape(y.size)  # make it plain
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain

            skip = False
            i_max = numpy.argmax(y)
            if i_max == labels[i]:
                n_ok += 1
                # check for threshold
                if y[i_max] >= self.threshold:
                    err_y[:] = 0  # already trained good enough, skip it
                    skip = True
            if not skip:
                # Compute softmax output error gradient
                err_y[:] = y[:]
                err_y[labels[i]] = y[labels[i]] - 1.0
        print("(n_ok, n_total): (%d, %d)" % (n_ok, batch_size))
        if n_ok == batch_size:
            print("Perfect")
            self.status.completed = True
            self.status.update()
            return

        t2 = time.time()
        err_y = self.err_y.batch
        print("Computed softmax errs within %.2f sec: (min, max, avg) = (%.3f, %.3f, %.3f)" % \
              (t2 - t1, err_y.min(), err_y.max(), numpy.average(err_y)))
    
        self.err_y.update()
