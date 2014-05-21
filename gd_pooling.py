"""
Created on Dec 3, 2013

Gradient Descent for Pooling units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import time

import veles.error as error
import veles.znicz.nn_units as nn_units


class GDPooling(nn_units.GradientDescentBase):
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
    def __init__(self, workflow, **kwargs):
        kx = kwargs.get("kx", 2)
        ky = kwargs.get("ky", 2)
        sliding = kwargs.get("sliding", (kx, ky))
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        kwargs["sliding"] = sliding
        super(GDPooling, self).__init__(workflow, **kwargs)
        self.kx = kx
        self.ky = ky
        self.sliding = sliding

    def init_unpickled(self):
        super(GDPooling, self).init_unpickled()
        self.cl_sources_["gradient_descent_pooling.cl"] = {}
        self.krn_err_input_ = None
        self.krn_err_input_clear_ = None

    def initialize(self, **kwargs):
        super(GDPooling, self).initialize(**kwargs)
        self._batch_size = self.input.mem.shape[0]
        self._sy = self.input.mem.shape[1]
        self._sx = self.input.mem.shape[2]
        self._n_channels = self.input.mem.size // (self._batch_size * self._sx *
                                                 self._sy)
        self._out_sx = self._sx // self.sliding[0] + (
            0 if self._sx % self.sliding[0] == 0 else 1)
        self._out_sy = self._sy // self.sliding[1] + (
            0 if self._sy % self.sliding[1] == 0 else 1)
        output_size = self._n_channels * self._out_sx * self._out_sy * \
            self._batch_size

        if self.err_output.mem.size != output_size:
            raise error.ErrBadFormat(
                "Size of err_output differs "
                "from the size computed based on kx, ky, size of input.")

        if (self.err_input.mem is None or
                self.err_input.mem.size != self.input.mem.size):
            self.err_input.reset()
            self.err_input.mem = numpy.zeros_like(self.input.mem)

        self.err_output.initialize(self.device)
        self.err_input.initialize(self.device)

        if self.device is None:
            return

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
                defines, "gd_pooling_%dx%dx%d_%dx%d.cl" %
                (self._sx, self._sy, self._n_channels,
                 self.kx, self.ky),
                dtype=self.err_output.mem.dtype)

        if self.krn_err_input_clear_ is None:
            self.krn_err_input_clear_ = self.get_kernel("array_clear")
            self.krn_err_input_clear_.set_arg(0, self.err_input.devmem)

    def print_debug_data(self, t_start):
        if not self.log.isEnabledFor(logging.DEBUG):
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
        event = self.execute_kernel([self.err_input.mem.size], None,
                                    self.krn_err_input_clear_)
        event.wait()

        # Compute err_h
        event = self.execute_kernel(
            [(self.batch_size or self.err_output.mem.shape[0]) *
             (self.err_output.mem.size // self.err_output.mem.shape[0])], None,
            self.krn_err_input_)
        event.wait()

    def cpu_run(self):
        raise error.ErrNotImplemented()

    def run(self):
        t1 = time.time()
        retval = super(GDPooling, self).run()
        if retval:
            return retval
        self.print_debug_data(t1)


class GDMaxPooling(GDPooling):
    """Gradient Descent for max pooling unit.

    Must be assigned before initialize():
        input_offs

    Updates after run():

    Creates within initialize():

    Attributes:
        input_offs: offsets in err_input where to copy err_output.
        krn_err_input_clear_: OpenCL kernel for setting err_input with zeros.
    """
    def __init__(self, workflow, **kwargs):
        super(GDMaxPooling, self).__init__(workflow, **kwargs)
        self.input_offs = None  # formats.Vector()

    def init_unpickled(self):
        super(GDMaxPooling, self).init_unpickled()
        self.krn_err_input_clear_ = None

    def initialize(self, **kwargs):
        super(GDMaxPooling, self).initialize(**kwargs)

        if self.err_output.mem.size != self.input_offs.mem.size:
            raise error.ErrBadFormat("Shape of err_output differs from "
                                     "that of input_offs")

        self.input_offs.initialize(self.device)

        if self.device is None:
            return

        if self.krn_err_input_ is None:
            self.krn_err_input_ = self.get_kernel("gd_max_pooling")
            self.krn_err_input_.set_args(self.err_output.devmem, self.err_input.devmem,
                                         self.input_offs.devmem)

    def ocl_run(self):
        """Do gradient descent on OpenCL device.
        """
        self.input_offs.unmap()  # we will use input_offs
        return super(GDMaxPooling, self).ocl_run()

    def cpu_run(self):
        """Do gradient descent on CPU.
        """
        self.err_output.map_read()
        self.input_offs.map_read()
        self.err_input.map_invalidate()
        self.err_input.mem[:] = 0

        if self.kx <= self.sliding[0] and self.ky <= self.sliding[1]:
            # self.input_offs cannot contain equal values - simple assignment
            for err, offset in numpy.nditer([self.err_output.mem,
                                             self.input_offs.mem]):
                batch, y, x, ch = numpy.unravel_index(offset,
                                                      self.err_input.mem.shape)
                self.err_input.mem[batch, y, x, ch] = err
        else:
            # self.input_offs can contain equal values
            for err, offset in numpy.nditer([self.err_output.mem,
                                             self.input_offs.mem]):
                batch, y, x, ch = numpy.unravel_index(offset,
                                                      self.err_input.mem.shape)
                self.err_input.mem[batch, y, x, ch] += err


class GDAvgPooling(GDPooling):
    """Gradient Descent for avg pooling unit.

    Must be assigned before initialize():

    Updates after run():

    Creates within initialize():

    """
    def initialize(self, **kwargs):
        super(GDAvgPooling, self).initialize(**kwargs)

        if self.device is None:
            return

        if self.krn_err_input_ is None:
            self.krn_err_input_ = self.get_kernel("gd_avg_pooling")
            self.krn_err_input_.set_args(self.err_output.devmem, self.err_input.devmem)

    def cpu_run(self):
        self.err_output.map_read()
        self.err_input.map_invalidate()

        if self.kx <= self.sliding[0] and self.ky <= self.sliding[1]:
            # disjoint kernels
            for (batch, y, x, ch), err in numpy.ndenumerate(self.err_output.mem):
                hx1 = x * self.sliding[0]
                hx2 = hx1 + self.kx
                hx2 = hx2 if hx2 < self._sx else self._sx
                hy1 = y * self.sliding[1]
                hy2 = hy1 + self.ky
                hy2 = hy2 if hy2 < self._sy else self._sy
                delta = err / float((hx2 - hx1) * (hy2 - hy1))
                for i, j in ((ii, jj) for ii in range(hy1, hy2)
                             for jj in range(hx1, hx2)):
                    self.err_input.mem[batch, i, j, ch] = delta
        else:
            # joint kernels
            self.err_input.mem[:] = 0
            for (batch, y, x, ch), err in numpy.ndenumerate(self.err_output.mem):
                hx1 = x * self.sliding[0]
                hx2 = hx1 + self.kx
                hx2 = hx2 if hx2 < self._sx else self._sx
                hy1 = y * self.sliding[1]
                hy2 = hy1 + self.ky
                hy2 = hy2 if hy2 < self._sy else self._sy
                delta = err / float((hx2 - hx1) * (hy2 - hy1))
                for i, j in ((ii, jj) for ii in range(hy1, hy2)
                             for jj in range(hx1, hx2)):
                    self.err_input.mem[batch, i, j, ch] += delta
