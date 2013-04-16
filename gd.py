"""
Created on Apr 15, 2013

Gradient Descent Filters.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import formats
import pyopencl
import numpy


class GDSM(filters.OpenCLFilter):
    """Gradient Descent for softmax.
    
    Attributes:
        weights: weights of the current layer.
        bias: bias of the current layer.
        y: outputs of the current layer.
        h: outputs of the hidden layer.
        labels: labels for y.
        err_y: dEds for y.
        err_h: dEds for h.
        weights_alpha: initial gradient descent speed.
        weights_lambda: coefficient for weights regularisation term ( lambda/2 * sum(weights^2) ).
        weights_alphas: own gradient speed for each weight.
    """
    def __init__(self, device = None, weights_alpha = 0.001, weights_lambda = 0.001, unpickling = 0):
        super(GDSM, self).__init__(device=device, unpickling=unpickling)
        if unpickling:
            return
        self.weights = None  # formats.Vector(device)
        self.bias = None  # formats.Vector(device)
        self.y = None  # formats.Batch(device)
        self.h = None  # formats.Batch(device)
        self.labels = None  # formats.Labels()
        self.err_y = formats.Batch(device)
        self.err_h = formats.Batch(device)
        self.weights_alpha = weights_alpha
        self.weights_lambda = weights_lambda
        self.weights_alphas = formats.Vector(device)

    def initialize(self):
        if self.err_h.batch == None or self.err_h.batch.size != self.h.batch.size:
            self.err_h.batch = filters.aligned_zeros(self.h.batch.shape)
            self.err_h.batch_ = None

        if self.err_y.batch == None or self.err_y.batch.size != self.y.batch.size:
            self.err_y.batch = filters.aligned_zeros(self.y.batch.shape)
            self.err_y.batch_ = None

        if self.weights_alphas.v == None or self.weights_alphas.v.size != self.weights.v.size:
            self.weights_alphas.v = filters.aligned_zeros(self.weights.v.shape)
            self.weights_alphas.v[:] = self.weights_alpha
            self.weights_alphas.v_ = None

        if not self.device:
            return

        self.err_h.initialize(self.device)
        self.err_y.initialize(self.device)
        self.weights_alphas.initialize(self.device)

    def cpu_run(self):
        n = self.y.batch.shape[0]
        labels = self.labels.batch
        for i in range(0, n):  # loop by batch
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            y = self.y.batch[i]
            err_y[:] = -y[:]
            err_y[labels[i]] = 1.0 - y[labels[i]]

            err_h = self.err_h.batch[i]
            #TODO(a.kazantsev): continue here.

    def run(self):
        return self.cpu_run()
