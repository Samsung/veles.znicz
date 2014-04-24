#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on April 23, 2013

A layer for local response normalization.
Detailed description given in article by Krizhevsky, Sutskever and Hinton:
"ImageNet Classification with Deep Convolutional Neural Networks"
"""

import numpy as np

from veles import units
from veles import formats

class LocalResponseNormalizer(units.Unit):
    """
    A base class for forward and backward units of local response normalization.
    """
    def __init__(self, workflow, **kwargs):
        self.alpha = kwargs.get("alpha", 0.0001)
        self.beta = kwargs.get("beta", 0.75)
        self.k = kwargs.get("k", 2)
        self.n = kwargs.get("n", 5)
        self.device = kwargs.get("device")

        super(LocalResponseNormalizer, self).__init__(workflow, **kwargs)

        self.input = None
        self.output = formats.Vector()

    def initialize(self):
        super(LocalResponseNormalizer, self).initialize()

class LRNormalizerForward(LocalResponseNormalizer):
    """
    Forward propagation of local response normalization.
    """

    def __init__(self, workflow, **kwargs):
        self.input = None  # input value of forward layer
        self.output = formats.Vector()  # output value of forward layer

        self.weights = None  # dummy attrs
        self.bias = None  # dummy attrs

        super(LRNormalizerForward, self).__init__(workflow, **kwargs)

    def initialize(self):
        super(LRNormalizerForward, self).initialize()
        self.output.v = np.ndarray(shape=self.input.v.shape,
                                   dtype=self.input.v.dtype)

    def run(self):
        """
        TODO
        """
        assert(len(self.input.v.shape) == 4)

        input_squared = np.square(self.input.v)
        subsums = np.ndarray(shape=self.input.v.shape, dtype=self.input.v.dtype)

        num_of_chans = self.input.v[3]

        for i in range(num_of_chans):
            min_range = max(0, i - int(self.n / 2))
            max_range = min(num_of_chans, i + int(self.n / 2))
            subsums[:, :, :, i] = np.sum(
                    input_squared[:, :, :, min_range:max_range], axis=3)
        subsums *= self.alpha
        subsums += self.k
        subsums **= self.beta

        np.copyto(self.output.v, self.input.v)
        self.output /= subsums


class LRNormalizerBackward(LocalResponseNormalizer):
    """
    Backward-propagation for local response normalization.
    """

    def __init__(self, workflow, **kwargs):
        self.y = None  # output of forward layer
        self.h = None  # input of forward layer
        self.err_y = None  # output error of fwd layer, our input error
        self.err_h = formats.Vector()  # input error of fwd layer, our output

        super(LRNormalizerBackward, self).__init__(workflow, **kwargs)


    def initialize(self):
        super(LRNormalizerBackward, self).initialize()
        self.err_h.v = np.ndarray(shape=self.err_y.v.shape,
                                   dtype=self.err_y.v.dtype)

    def run(self):
        # TODO: implementation
        np.copyto(self.err_h.v, self.err_y.v)
