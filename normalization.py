#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on April 23, 2013

A layer for local response normalization.
Detailed description given in article by Krizhevsky, Sutskever and Hinton:
"ImageNet Classification with Deep Convolutional Neural Networks"
"""

import numpy as np

from veles.znicz import nn_units
from veles import formats


class LocalResponseNormalizer(nn_units.Forward):
    """
    A base class for forward and backward units of local
    response normalization.
    """
    def __init__(self, workflow, **kwargs):
        self.alpha = kwargs.get("alpha", 0.0001)
        self.beta = kwargs.get("beta", 0.75)
        self.k = kwargs.get("k", 2)
        self.n = kwargs.get("n", 5)
        self._num_of_chans = None
        self.device = kwargs.get("device")

        super(LocalResponseNormalizer, self).__init__(workflow, **kwargs)

    def initialize(self, device, **kwargs):
        super(LocalResponseNormalizer, self).initialize(device, **kwargs)

    def _subsums(self, source_array, window_size):
        """
        For each channel calculates the sum of its neighbour channels.
        source_array must be a 4-dimensional array (channel dim is the last).
        """
        assert len(source_array.shape) == 4
        subsums = np.ndarray(shape=source_array.shape, dtype=np.float64)
        num_of_chans = source_array.shape[3]
        for i in range(num_of_chans):
            min_index = max(0, i - int(window_size / 2))
            max_index = min(i + int(window_size / 2), num_of_chans - 1)
            array_slice = source_array[:, :, :, min_index: max_index + 1]
            subsums[:, :, :, i] = np.sum(array_slice, axis=3)
        return subsums


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

    def init_unpickled(self):
        super(LRNormalizerForward, self).init_unpickled()
        self.cl_sources_["normalization.cl"] = {}

    def initialize(self, device, **kwargs):
        super(LRNormalizerForward, self).initialize(device, **kwargs)
        self.output.mem = np.ndarray(shape=self.input.mem.shape,
                                     dtype=self.input.mem.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)

        self._num_of_chans = self.input.mem.shape[3]

        defines = {"ALPHA": self.alpha, "BETA": self.beta, "K": self.k,
                   "N": self.n, "NUM_OF_CHANS": self._num_of_chans}

        self.build_program(defines, dtype=self.input.mem.dtype)
        self.assign_kernel("forward")
        self.set_args(self.input, self.output)

        self._global_size_ = [self.output.mem.size // self._num_of_chans]
        self._local_size_ = None

    def cpu_run(self):
        self.output.map_invalidate()
        self.input.map_read()

        assert len(self.input.mem.shape) == 4
        input_squared = np.square(self.input.mem)
        subsums = self._subsums(input_squared, self.n)
        subsums *= self.alpha
        subsums += self.k
        subsums **= self.beta

        np.copyto(self.output.mem, self.input.mem / subsums)

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()
        self.input.unmap()
        self.execute_kernel(self._global_size_, self._local_size_).wait()

    def run(self):
        self.ocl_run()


class LRNormalizerBackward(LocalResponseNormalizer):
    """
    Backward-propagation for local response normalization.
    """
    def __init__(self, workflow, **kwargs):
        self.output = None  # output of forward layer
        self.input = None  # input of forward layer
        self.err_output = None  # output error of fwd unit, our input error
        self.err_input = formats.Vector()  # in error of fwd unit, our output

        super(LRNormalizerBackward, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(LRNormalizerBackward, self).init_unpickled()
        self.cl_sources_["normalization.cl"] = {}

    def initialize(self, device, **kwargs):
        super(LRNormalizerBackward, self).initialize(device, **kwargs)
        self.err_input.mem = np.zeros(self.err_output.mem.shape,
                                      dtype=self.err_output.mem.dtype)

        self.err_output.initialize(self.device)
        self.input.initialize(self.device)
        self.err_input.initialize(self.device)

        self._num_of_chans = self.input.mem.shape[3]

        defines = {"ALPHA": self.alpha, "BETA": self.beta, "K": self.k,
                   "N": self.n, "NUM_OF_CHANS": self._num_of_chans}

        self.build_program(defines, dtype=self.input.mem.dtype)
        self.assign_kernel("backward")
        self.set_args(self.err_output, self.input, self.err_input)

        self._global_size_ = [self.err_input.mem.size // self._num_of_chans]
        self._local_size_ = None

    def cpu_run(self):
        self.err_input.map_invalidate()
        self.err_output.map_read()
        self.input.map_read()

        assert len(self.input.mem.shape) == 4
        assert self.input.mem.shape == self.err_output.mem.shape

        num_of_chans = self.input.mem.shape[3]

        input_squared = np.square(self.input.mem)
        input_subsums = self._subsums(input_squared, self.n)

        input_subsums *= self.alpha
        input_subsums += self.k

        input_subsums_powered = np.power(input_subsums, (self.beta + 1))

        err_h = self.err_input.mem
        err_y = self.err_output.mem

        for i in range(num_of_chans):
            min_index = max(0, i - int(self.n / 2))
            max_index = min(i + int(self.n / 2), num_of_chans - 1)

            delta_h = np.zeros(dtype=np.float64,
                               shape=err_h[:, :, :, i].shape)
            for j in range(min_index, max_index + 1):
                dh = np.zeros(shape=delta_h.shape, dtype=np.float64)
                if i == j:
                    dh += input_subsums[:, :, :, j]
                dh -= (2 * self.beta * self.alpha *
                       self.input.mem[:, :, :, i] *
                       self.input.mem[:, :, :, j])
                dh *= (err_y[:, :, :, j] /
                       input_subsums_powered[:, :, :, j])
                delta_h += dh
            np.copyto(err_h[:, :, :, i], delta_h)

    def ocl_run(self):
        self.err_output.unmap()
        self.input.unmap()
        self.err_input.unmap()
        self.execute_kernel(self._global_size_, self._local_size_).wait()

    def run(self):
        self.ocl_run()
