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
        self.demand("input", "err_output")

    def init_unpickled(self):
        super(GDPooling, self).init_unpickled()
        self.cl_sources_["gradient_descent_pooling"] = {}
        self.krn_err_input_ = None
        self.krn_err_input_clear_ = None

    def initialize(self, device, **kwargs):
        self.create_output()
        if self.err_output.size != self._output_size:
            raise error.BadFormatError(
                "Size of err_output differs "
                "from the size computed based on kx, ky, size of input.")
        super(GDPooling, self).initialize(device=device, **kwargs)

    def ocl_init(self):
        if self.program_ is None:
            defines = {
                'SX': self._sx,
                'SY': self._sy,
                'N_CHANNELS': self._n_channels,
                'KX': self.kx,
                'KY': self.ky,
                'SLIDE_X': self.sliding[0],
                'SLIDE_Y': self.sliding[1]
            }
            self.build_program(
                defines, "%s_%d_%dx%dx%d_%dx%d" %
                (self.__class__.__name__, self.err_output.shape[0],
                 self._sx, self._sy, self._n_channels,
                 self.kx, self.ky),
                dtype=self.err_output.mem.dtype)

        if self.krn_err_input_clear_ is None:
            self.krn_err_input_clear_ = self.get_kernel("err_input_clear")
            self.krn_err_input_clear_.set_arg(0, self.err_input.devmem)

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
        self.err_input.unmap()  # we will update err_input
        self.err_output.unmap()  # we will use err_output

        # Clear err_h
        self.execute_kernel([self.err_input.mem.size], None,
                            self.krn_err_input_clear_)

        # Compute err_h
        self.execute_kernel(
            [(self.batch_size or self.err_output.mem.shape[0]) *
             (self.err_output.mem.size // self.err_output.mem.shape[0])], None,
            self.krn_err_input_)

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

    def init_unpickled(self):
        super(GDMaxPooling, self).init_unpickled()
        self.krn_err_input_clear_ = None

    def initialize(self, device, **kwargs):
        super(GDMaxPooling, self).initialize(device=device, **kwargs)

        if self.err_output.size != self.input_offset.size:
            raise error.BadFormatError("Shape of err_output differs from "
                                       "that of input_offset")

        self.input_offset.initialize(self.device)

        if self.device is None:
            return

        if self.krn_err_input_ is None:
            self.krn_err_input_ = self.get_kernel("gd_max_pooling")
            self.krn_err_input_.set_args(self.err_output.devmem,
                                         self.err_input.devmem,
                                         self.input_offset.devmem)

    # IDistributable implementation
    def generate_data_for_slave(self, slave):
        self.input_offset.map_read()
        data = (self.input_offset.mem)
        return data

    def apply_data_from_master(self, data):
        self.input_offset.map_invalidate()
        self.input_offset.mem[:] = data[0][:]

    def ocl_run(self):
        """Do gradient descent on OpenCL device.
        """
        self.input_offset.unmap()  # we will use input_offset
        return super(GDMaxPooling, self).ocl_run()

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
                                                  self.err_input.mem.shape)
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
        super(GDAvgPooling, self).initialize(device=device, **kwargs)

        if self.device is None:
            return

        if self.krn_err_input_ is None:
            self.krn_err_input_ = self.get_kernel("gd_avg_pooling")
            self.krn_err_input_.set_args(self.err_output.devmem,
                                         self.err_input.devmem)

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
