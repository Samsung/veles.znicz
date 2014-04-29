"""
Created on Apr 25, 2014

A dropout layer. It is a signal repeater with some repeating channels set to 0.
Inputs to be disabled are randomly selected each forward proparation.

Detailed description given in article by Krizhevsky, Sutskever and Hinton:
"ImageNet Classification with Deep Convolutional Neural Networks" (sec. 4.2).
"""

import numpy as np

from veles import OpenCLUnit
import veles.opencl_types as opencl_types
from veles import formats


class Dropout(OpenCLUnit):
    """
    A base class for forward and backward units of local
    response normalization.
    """
    def __init__(self, workflow, **kwargs):
        super(Dropout, self).__init__(workflow, **kwargs)
        self.dropout_ratio = kwargs.get("dropout_ratio", 0.5)

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
        self.output = None  # output value of forward layer

        super(DropoutForward, self).__init__(workflow, **kwargs)

    @Dropout.dropout_ratio.setter
    def dropout_ratio(self, value):
        Dropout.dropout_ratio.fset(self, value)
        if self.input is not None:
            self.calc_weights()

    def initialize(self, device, **kwargs):
        super(DropoutForward, self).initialize(device=device, **kwargs)
        self.calc_weights()
        output_size = int(self.input.v.size // self.input.v.shape[0])
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)]
        self._global_size_ = [formats.roundup(output_size, block_size),
                              formats.roundup(self.input.v.shape[0],
                                              block_size)]
        self._local_size_ = [block_size, block_size]

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

    def ocl_run(self):
        self.execute_kernel(self.krn_, self._global_size_,
                            self._local_size_).wait()


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
        """
        output_size = int(self.output.v.size // self.output.v.shape[0])
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)]
        self._global_size_ = [formats.roundup(output_size, block_size),
                              formats.roundup(self.output.v.shape[0],
                                              block_size)]
        self._local_size_ = [block_size, block_size]
        """

    def cpu_run(self):
        self.err_h.map_invalidate()
        self.err_y.map_read()
        self.weights.map_read()

        self.err_h.v = self.err_y.v * self.weights.v

    def ocl_run(self):
        self.execute_kernel(self.krn_).wait()
