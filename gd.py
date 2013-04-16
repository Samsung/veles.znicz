"""
Created on Apr 15, 2013

Gradient Descent Filters.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import formats
import pyopencl
import numpy
import time


class GDSM(filters.OpenCLFilter):
    """Gradient Descent for softmax.
    
    Attributes:
        weights: weights of the current layer.
        bias: bias of the current layer.
        y: outputs of the current layer.
        h: outputs of the hidden layer.
        labels: labels for y.
        err_y: backpropagation errors for y.
        err_h: backpropagation errors for h (will compute them).
        global_alpha: gradient descent speed (positive).
        global_lambda: coefficient (positive or zero) for weights regularization term (lambda/2 * sum(weights^2)).
    """
    def __init__(self, device = None, global_alpha = 0.1, global_lambda = 0.01, unpickling = 0):
        super(GDSM, self).__init__(device=device, unpickling=unpickling)
        if unpickling:
            return
        self.weights = None  # formats.Vector(device)
        self.bias = None  # formats.Vector(device)
        self.y = None  # formats.Batch(device)
        self.h = None  # formats.Batch(device)
        self.err_y = None  # formats.Batch(device)
        self.err_h = formats.Batch(device)
        self.global_alpha = global_alpha
        self.global_lambda = global_lambda

    def initialize(self):
        if self.err_h.batch == None or self.err_h.batch.size != self.h.batch.size:
            self.err_h.batch = filters.aligned_zeros(self.h.batch.shape)
            self.err_h.batch_ = None

        if not self.device:
            return

        self.err_h.initialize(self.device)

    def cpu_run(self):
        """Do gradient descent in case of softmax activation.
        """
        t1 = time.time()

        batch_size = self.y.batch.shape[0]
        r_batch_size = 1.0 / batch_size
        weights = self.weights.v.transpose()
        for i in range(0, batch_size):  # loop by batch
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            err_h = self.err_h.batch[i]
            err_h = err_h.reshape(err_h.size)  # make it plain
            numpy.dot(weights, err_y, err_h)

        # Weights update
        bias = self.bias.v
        for i in range(0, batch_size):  # loop by batch
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            h = self.h.batch[i]
            h = h.reshape(h.size)  # make it plain
            weights += numpy.outer(h, err_y) * (self.global_alpha * r_batch_size)
            bias += err_y * (self.global_alpha * r_batch_size)
        weights *= 1.0 + self.global_lambda  # regularization (will not regularize bias)

        t2 = time.time()
        print("Backprop within %.2f sec: (min, max, avg) = (%.6f, %.6f, %.6f)" % \
              (t2 - t1, weights.min(), weights.max(), numpy.average(weights))) 

        self.weights.update()
        self.bias.update()
        self.err_h.update()
    