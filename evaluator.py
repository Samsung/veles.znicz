"""
Created on Apr 1, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import formats
import numpy
import time
#import matplotlib.pyplot as pp
#import matplotlib.cm as cm


class BatchEvaluator(filters.OpenCLFilter):
    """Evaluator for nn softmax output from the batch labels.

    Attributes:
        labels: labels for Batch.
        y: output of the network as Batch.
        err_y: backpropagation errors based on labels.
        status: status of the evaluation (status.completed = True when learning ended).
        threshold: threshold for skipping trained well enough samples.
    """
    def __init__(self, threshold = 0.25, device = None, unpickling = 0):
        super(BatchEvaluator, self).__init__(unpickling=unpickling, device=device)
        self.save_failed = False
        self.first_run = True
        if unpickling:
            return
        self.labels = None  # formats.Labels()
        self.y = None  # formats.Batch(device)
        self.err_y = formats.Batch()
        self.status = filters.Connector()
        self.status.completed = False
        self.status.n_ok = 0
        self.threshold = threshold

    def initialize(self):
        if self.err_y.batch == None or self.err_y.batch.size != self.y.batch.size:
            self.err_y.batch = filters.aligned_zeros(self.y.batch.shape)
            self.err_y.batch_ = None

        self.err_y.initialize(self.device)

    def cpu_run(self):
        t1 = time.time()

        self.y.sync()
        n_ok = 0
        n_skip = 0
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
                    n_skip += 1
                if self.save_failed and (i % 50) == 0:
                    idx = numpy.argsort(y)
                    pp.imshow(self.origin.batch[i], interpolation="lanczos", cmap=cm.gray)
                    width = 256
                    fnme = "ok/%d_as_%d(%d)_%d(%d)_%d(%d).%d.png" % (labels[i], i_max, y[i_max] * 100, \
                                                                     idx[y.size - 2], y[idx[y.size - 2]] * 100, \
                                                                     idx[y.size - 3], y[idx[y.size - 3]] * 100, \
                                                                     i)
                    pp.savefig(fnme, dpi=width//8)
                    print("Image saved to %s" % (fnme, ))
                    pp.clf()
                    pp.cla()
            elif self.save_failed:
                idx = numpy.argsort(y)
                pp.imshow(self.origin.batch[i], interpolation="lanczos", cmap=cm.gray)
                width = 256
                fnme = "failed/%d_as_%d(%d)_%d(%d)_%d(%d).%d.png" % (labels[i], i_max, y[i_max] * 100, \
                                                                     idx[y.size - 2], y[idx[y.size - 2]] * 100, \
                                                                     idx[y.size - 3], y[idx[y.size - 3]] * 100, \
                                                                     i)
                pp.savefig(fnme, dpi=width//8)
                print("Image saved to %s" % (fnme, ))
                pp.clf()
                pp.cla()

            if not skip:
                # Compute softmax output error gradient
                err_y[:] = y[:]
                err_y[labels[i]] = y[labels[i]] - 1.0
        self.err_y.update()
        self.status.n_ok = n_ok
        self.status.completed = False
        print("(n_ok, n_total): (%d, %d)" % (n_ok, batch_size))
        if not self.first_run and (self.threshold == 1.0 or n_skip == batch_size) and n_ok == batch_size:
            print("Perfect")
            self.status.completed = True
            self.status.update()
            return
        self.first_run = False

        dt = time.time() - t1
        if not __debug__:
            print("Computed softmax errs within %.2f sec, skipped %.2f%%" % (dt, n_skip / batch_size * 100.0))
            return
        err_y = self.err_y.batch
        print("Computed softmax errs within %.2f sec, skipped %.2f%%: (min, max, avg) = (%.3f, %.3f, %.3f)" % \
              (dt, n_skip / batch_size * 100.0, err_y.min(), err_y.max(), numpy.average(err_y)))
