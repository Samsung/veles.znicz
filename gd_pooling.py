"""
Created on Dec 3, 2013

Gradient descent units for **pooling** layers.

* :class:`GDMaxPooling` couples with :class:`veles.znicz.pooling.MaxPooling`
* :class:`GDAvgPooling` couples with :class:`veles.znicz.pooling.AvgPooling`
* :class:`GDMaxAbsPooling` couples with \
    :class:`veles.znicz.pooling.MaxAbsPooling`


Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import logging
import numpy
import time
from zope.interface import implementer

import veles.error as error
from veles.accelerated_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units
from veles.distributable import TriviallyDistributable
from veles.znicz.pooling import PoolingBase


@implementer(IOpenCLUnit)
class GDPooling(PoolingBase, nn_units.GradientDescentBase,
                TriviallyDistributable):
    """Gradient Descent for pooling unit.

    Must be assigned before initialize():
        err_output
        input

    Updates after run():
        err_input

    Creates within initialize():
        err_input

    Attributes:
        kx: pooling kernel width.
        ky: pooling kernel height.
        sliding: tuple of kernel sliding (by x-axis, by y-axis).
        err_output: backpropagation errors for output.
        input: input (will get only shape from it).
        err_input: backpropagation errors for input (will compute its).
        krn_err_input_: OpenCL kernel for computing err_input.
    """
    MAPPING = set()

    def __init__(self, workflow, **kwargs):
        super(GDPooling, self).__init__(workflow, **kwargs)
        self.kernel_name = None
        self.demand("input", "err_output")

    def init_unpickled(self):
        super(GDPooling, self).init_unpickled()
        self.sources_["gradient_descent_pooling"] = {}
        self.krn_err_input_ = None
        self.krn_err_input_clear_ = None

    def initialize(self, device, **kwargs):
        self.create_output()
        if self.err_output.size != self._output_size:
            raise error.BadFormatError(
                "Size of err_output differs "
                "from the size computed based on kx, ky, size of input.")
        super(GDPooling, self).initialize(device=device, **kwargs)

    def _gpu_init(self):
        defines = {
            'SX': self._sx,
            'SY': self._sy,
            'N_CHANNELS': self._n_channels,
            'KX': self.kx,
            'KY': self.ky,
            'SLIDE_X': self.sliding[0],
            'SLIDE_Y': self.sliding[1],
            'OUTPUT_SIZE': self.err_output.size
        }
        self.build_program(
            defines, "%s_%d_%dx%dx%d_%dx%d" %
            (self.__class__.__name__, self.err_output.shape[0],
             self._sx, self._sy, self._n_channels,
             self.kx, self.ky),
            dtype=self.err_output.dtype)
        self.assign_kernel(self.kernel_name)
        self.set_args(self.err_output, self.err_input)

    def ocl_init(self):
        self._gpu_init()
        if self.krn_err_input_clear_ is None:
            self.krn_err_input_clear_ = self.get_kernel("err_input_clear")
            self.krn_err_input_clear_.set_arg(0, self.err_input.devmem)

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = lambda: (
            int(numpy.ceil(self.current_batch_size *
                           self.err_output.sample_size / block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def print_debug_data(self, t_start):
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        output = self.err_input.mem
        self.debug(
            "%s: %d samples of size %dx%dx%d and sliding %dx%d in %.2f sec" % (
                self.__class__.__name__,
                output.shape[0], output.shape[2], output.shape[1],
                output.shape[3], self.sliding[0], self.sliding[1],
                time.time() - t_start))

    def ocl_run(self):
        """Do gradient descent.
        """
        self.unmap_vectors(self.err_input, self.err_output)

        # Clear err_h
        self.execute_kernel([self.err_input.size], None,
                            self.krn_err_input_clear_)

        # Compute err_h
        self.execute_kernel(
            [self.current_batch_size * self.err_output.sample_size], None)

    def cuda_run(self):
        self.unmap_vectors(self.err_input, self.err_output)

        # Clear err_input
        self.err_input.devmem.memset32_async()

        # Compute err_input
        self.execute_kernel(self._global_size(), self._local_size)

    def cpu_run(self):
        raise NotImplementedError()

    def run(self):
        t1 = time.time()
        retval = super(GDPooling, self).run()
        if retval:
            return retval
        self.print_debug_data(t1)


class GDMaxPooling(GDPooling):
    """Gradient Descent for max pooling unit.

    Must be assigned before initialize():
        input_offset

    Updates after run():

    Creates within initialize():

    Attributes:
        input_offset: offsets in err_input where to copy err_output.
        krn_err_input_clear_: OpenCL kernel for setting err_input with zeros.
    """

    MAPPING = {"max_pooling", "stochastic_pooling", "stochastic_pool_depool",
               "stochastic_abs_pool_depool"}

    def __init__(self, workflow, **kwargs):
        super(GDMaxPooling, self).__init__(workflow, **kwargs)
        self.input_offset = None  # formats.Vector()
        self.demand("input_offset")

    def initialize(self, device, **kwargs):
        self.kernel_name = "gd_max_pooling"
        super(GDMaxPooling, self).initialize(device=device, **kwargs)

        if self.err_output.size != self.input_offset.size:
            raise error.BadFormatError("Shape of err_output differs from "
                                       "that of input_offset")

        self.input_offset.initialize(self.device)

    def ocl_init(self):
        super(GDMaxPooling, self).ocl_init()
        self.set_arg(2, self.input_offset)

    def cuda_init(self):
        super(GDMaxPooling, self).cuda_init()
        self.set_arg(2, self.input_offset)

    def ocl_run(self):
        """Do gradient descent on OpenCL device.
        """
        self.input_offset.unmap()  # we will use input_offset
        return super(GDMaxPooling, self).ocl_run()

    def cuda_run(self):
        self.input_offset.unmap()
        return super(GDMaxPooling, self).cuda_run()

    def cpu_run(self):
        """Do gradient descent on CPU.
        """
        self.err_output.map_read()
        self.input_offset.map_read()
        self.err_input.map_invalidate()
        self.err_input.mem[:] = 0

        # self.input_offset can contain equal values
        for err, offset in numpy.nditer([self.err_output.mem,
                                         self.input_offset.mem]):
            batch, y, x, ch = numpy.unravel_index(offset,
                                                  self.err_input.shape)
            self.err_input.mem[batch, y, x, ch] += err


class GDMaxAbsPooling(GDMaxPooling):
    """Gradient descent is the same as in GDMaxPooling.
    """
    MAPPING = {"maxabs_pooling", "stochastic_abs_pooling"}


class GDAvgPooling(GDPooling):
    """Gradient Descent for avg pooling unit.

    Must be assigned before initialize():

    Updates after run():

    Creates within initialize():

    """

    MAPPING = {"avg_pooling"}

    def initialize(self, device, **kwargs):
        self.kernel_name = "gd_avg_pooling"
        super(GDAvgPooling, self).initialize(device=device, **kwargs)

    def cpu_run(self):
        self.err_output.map_read()
        self.err_input.map_invalidate()
        self.err_input.mem[:] = 0

        for (batch, y, x, ch), err in numpy.ndenumerate(self.err_output.mem):
            hx1 = x * self.sliding[0]
            hx2 = hx1 + self.kx
            hx2 = hx2 if hx2 < self._sx else self._sx
            hy1 = y * self.sliding[1]
            hy2 = hy1 + self.ky
            hy2 = hy2 if hy2 < self._sy else self._sy
            delta = err / ((hx2 - hx1) * (hy2 - hy1))
            for i, j in ((ii, jj) for ii in range(hy1, hy2)
                         for jj in range(hx1, hx2)):
                self.err_input.mem[batch, i, j, ch] += delta
