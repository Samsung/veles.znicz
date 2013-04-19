"""
Created on Apr 15, 2013

Gradient Descent Filters.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import formats
import numpy
import time


class GD(filters.OpenCLFilter):
    """Gradient Descent.

    Attributes:
        weights: weights of the current layer.
        bias: bias of the current layer.
        y: outputs of the current layer.
        h: outputs of the hidden layer.
        err_y: backpropagation errors for y.
        err_h: backpropagation errors for h (will compute its).
        global_alpha: gradient descent speed (positive).
        global_lambda: coefficient (positive or zero) for weights regularization term (lambda/2 * sum(weights^2)).
    """
    def __init__(self, device = None, global_alpha = 0.1, global_lambda = 0.01, unpickling = 0):
        super(GD, self).__init__(device=device, unpickling=unpickling)
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

    def cpu_weights_update(self):
        self.h.sync()
        self.bias.sync()
        bias = self.bias.v
        batch_size = self.y.batch.shape[0]
        r_batch_size = 1.0 / batch_size
        weights = self.weights.v.transpose()
        weights *= 1.0 + ((-self.global_alpha) * self.global_lambda)  # regularization (will not regularize bias)
        for i in range(0, batch_size):  # loop by batch
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            h = self.h.batch[i]
            h = h.reshape(h.size)  # make it plain
            weights += numpy.outer(h, err_y) * ((-self.global_alpha) * r_batch_size)
            bias += err_y * ((-self.global_alpha) * r_batch_size)
        self.weights.update()
        self.bias.update()

    def print_times(self, t_start):
        weights = self.weights.v
        bias = self.bias.v
        print("Backprop within %.2f sec: (W, b) = (%.6f, %.6f), (%.6f, %.6f)" % \
              (time.time() - t_start, weights.min(), weights.max(), bias.min(), bias.max()))


class GDSM(GD):
    """Gradient Descent for softmax.
    """
    def cpu_run(self):
        """Do gradient descent in case of softmax activation.
        """
        t1 = time.time()
        # Backpropagate error (will compute err_h)
        self.y.sync()
        self.err_y.sync()
        self.weights.sync()
        batch_size = self.y.batch.shape[0]
        weights = self.weights.v.transpose()
        for i in range(0, batch_size):  # loop by batch
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            err_h = self.err_h.batch[i]
            err_h = err_h.reshape(err_h.size)  # make it plain
            numpy.dot(weights, err_y, err_h)
        self.err_h.update()
        # Update weights
        self.cpu_weights_update()
        self.print_times(t1)


class GDTanh(GD):
    """Gradient Descent for f(): y = 1.7159 * tanh(0.6666 * (W * x + b)).

    f'(y) = (a * tanh(b * y))' = a * (1 - b^2 * y^2) * b = a * b - a * b^3 * y^2
          = 1.143819 - 0.508262 * y^2
    """
    def cpu_run(self):
        """Do gradient descent in case of softmax activation.
        """
        t1 = time.time()
        # Backpropagate error (will compute err_h)
        self.y.sync()
        self.err_y.sync()
        self.weights.sync()
        batch_size = self.y.batch.shape[0]
        weights = self.weights.v.transpose()
        for i in range(0, batch_size):  # loop by batch
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            y = self.y.batch[i]
            y = y.reshape(y.size)  # make it plain
            err_y *= y * y * (-0.508262) + 1.143819
            err_h = self.err_h.batch[i]
            err_h = err_h.reshape(err_h.size)  # make it plain
            numpy.dot(weights, err_y, err_h)
        self.err_y.update()
        # Update weights
        self.cpu_weights_update()
        self.print_times(t1)
