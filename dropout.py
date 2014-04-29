"""
Created on Apr 25, 2014

A dropout layer. It is a signal repeater with some repeating channels set to 0.
Inputs to be disabled are randomly selected each forward proparation.

Detailed description given in article by Krizhevsky, Sutskever and Hinton:
"ImageNet Classification with Deep Convolutional Neural Networks" (sec. 4.2).
"""

import numpy as np

from veles import OpenCLUnit
from veles import formats


class Dropout(OpenCLUnit):
    """
    A base class for forward and backward units of local
    response normalization.
    """
    def __init__(self, workflow, **kwargs):
        # what fraction of inputs to disable
        self.dropout_ratio = kwargs.get("dropout_ratio", 0.5)
        assert 0. < self.dropout_ratio < 1.
        self.device = kwargs.get("device")
        super(Dropout, self).__init__(workflow, **kwargs)


class DropoutForward(Dropout):
    """
    Forward propagation of dropout layer.
    """
    def __init__(self, workflow, **kwargs):
        self.input = None  # input value of forward layer
        self.output = formats.Vector()  # output value of forward layer

        self.weights = formats.Vector()  # dropout mask
        self.bias = None  # dummy attrs

        super(DropoutForward, self).__init__(workflow, **kwargs)

    def initialize(self, device, **kwargs):
        super(DropoutForward, self).initialize(device=device, **kwargs)
        self.calc_weights()

    def calc_weights(self):
        leave_ratio = 1.0 - self.dropout_ratio
        self.weights.v = np.random.uniform(low=-self.dropout_ratio,
                                           high=leave_ratio,
                                           size=self.input.v.shape)
        self.weights.v = np.maximum(self.weights.v, 0) / leave_ratio

    def cpu_run(self):
        self.output.map_invalidate()
        self.weights.map_invalidate()
        self.input.map_read()

        self.output.v = self.input.v * self.weights.v
        self.calc_weights()


class DropoutBackward(Dropout):
    """
    Backward propagation of droupout layer.
    """
    def __init__(self, workflow, **kwargs):
        self.y = None  # output of forward layer
        self.h = None  # input of forward layer
        self.weights = None  # dropout mask (should be given from forward unit)
        self.err_y = None  # output error of fwd layer, our input error
        self.err_h = formats.Vector()  # input error of fwd layer, our output

        super(DropoutBackward, self).__init__(workflow, **kwargs)

    def initialize(self, device, **kwargs):
        super(DropoutBackward, self).initialize(device=device, **kwargs)
        self.err_h.v = np.zeros(shape=self.err_y.v.shape,
                                dtype=self.err_y.v.dtype)
        assert self.h.v.shape == self.y.v.shape == self.err_y.v.shape

    def run(self):
        self.err_h.map_invalidate()
        self.err_y.map_read()
        self.weights.map_read()

        self.err_h.v = self.err_y.v * self.weights.v
