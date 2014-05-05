"""
Created on Apr 25, 2014

A dropout layer. It is a signal repeater with some repeating channels set to 0.
Inputs to be disabled are randomly selected each forward proparation.

Detailed description given in article by Krizhevsky, Sutskever and Hinton:
"ImageNet Classification with Deep Convolutional Neural Networks" (sec. 4.2).
"""

import numpy as np

from veles import config, formats, OpenCLUnit
import veles.rnd as rnd


class Dropout(OpenCLUnit):
    """
    A base class for forward and backward units of local
    response normalization.
    """
    def __init__(self, workflow, **kwargs):
        super(Dropout, self).__init__(workflow, **kwargs)
        self.dropout_ratio = kwargs.get("dropout_ratio", 0.5)

    def init_unpickled(self):
        super(Dropout, self).init_unpickled()
        self.cl_sources_["dropout.cl"] = {}

    @property
    def dropout_ratio(self):
        """ Gets the relative amount of weights to disable.
        """
        return self._dropout_ratio

    @dropout_ratio.setter
    def dropout_ratio(self, value):
        """ Sets the relative amount of weights to disable.
        """
        assert 0. < value < 1.
        self._dropout_ratio = value


class DropoutForward(Dropout):
    """
    Forward propagation of dropout layer.
    """
    def __init__(self, workflow, **kwargs):
        self.input = None  # input value of forward layer
        self.weights = formats.Vector()  # dropout mask
        self.states = formats.Vector()
        self.rnd = kwargs.get("rnd", rnd.default)
        super(DropoutForward, self).__init__(workflow, **kwargs)

    @Dropout.dropout_ratio.setter
    def dropout_ratio(self, value):
        Dropout.dropout_ratio.fset(self, value)
        if self.input is not None:
            self.calc_weights()

    @property
    def output(self):
        return self.input

    def initialize(self, device, **kwargs):
        super(DropoutForward, self).initialize(device=device, **kwargs)
        self.calc_weights()
        self.states.v = self.rnd.randint(
            low=0, high=0x100000000,
            size=self.input.v.size * 4).astype(np.uint32)
        self.input.initialize(device)
        self.states.initialize(device)
        self.weights.initialize(device)
        self._threshold_arg_ = np.empty(1, dtype=np.uint64)
        self._pass_arg_ = np.empty(1, dtype=self.input.v.dtype)
        sample_size = self.input.v.size // self.input.v.shape[0]

        self.build_program(
            {}, "%s/dropout_forward_%d.cl" %
            (config.root.common.cache_dir, sample_size),
            dtype=self.input.v.dtype)

        self.krn_ = self.get_kernel("dropout_forward")
        self.krn_.set_arg(0, self.input.v_)
        self.krn_.set_arg(3, self.states.v_)
        self.krn_.set_arg(4, self.weights.v_)
        self.krn_.set_arg(5, self.output.v_)

    def calc_weights(self):
        leave_ratio = 1.0 - self.dropout_ratio
        self.weights.v = np.random.uniform(low=-self.dropout_ratio,
                                           high=leave_ratio,
                                           size=self.input.v.size)
        np.maximum(self.weights.v, 0, self.weights.v)
        np.ceil(self.weights.v, self.weights.v)
        self.weights.v = (self.weights.v.astype(self.input.v.dtype) /
                          leave_ratio)

    def cpu_run(self):
        self.output.map_invalidate()
        self.weights.map_invalidate()
        self.input.map_read()

        self.output.v = self.input.v * self.weights.v
        self.calc_weights()

    def ocl_run(self):
        self.input.unmap()
        self.states.unmap()
        self.weights.unmap()
        self.output.unmap()
        self._threshold_arg_[0] = ((1 << 64) + 0.) * self.dropout_ratio
        self._pass_arg_[0] = 1.0 / (1.0 - self.dropout_ratio)
        self.krn_.set_arg(1, self._threshold_arg_)
        self.krn_.set_arg(2, self._pass_arg_)
        self.execute_kernel(self.krn_, (self.input.v.size,), None).wait()


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

    def cpu_run(self):
        self.err_h.map_invalidate()
        self.err_y.map_read()
        self.weights.map_read()
        np.multiply(self.err_y.v.ravel(), self.weights.v.ravel(),
                    formats.ravel(self.err_h.v))

    def ocl_run(self):
        return self.cpu_run()
    # TODO: implement backward propagation kernel
