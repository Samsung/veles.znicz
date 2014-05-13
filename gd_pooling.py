"""
Created on Dec 3, 2013

Gradient Descent for Pooling units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import time

from veles.config import root
import veles.error as error
import veles.znicz.nn_units as nn_units


class GDPooling(nn_units.GradientDescentBase):
    """Gradient Descent for pooling unit.

    Should be assigned before initialize():
        err_y
        h

    Updates after run():
        err_h

    Creates within initialize():
        err_h

    Attributes:
        kx: pooling kernel width.
        ky: pooling kernel height.
        sliding: tuple of kernel sliding (by x-axis, by y-axis).
        err_y: backpropagation errors for y.
        h: input (will get only shape from it).
        err_h: backpropagation errors for h (will compute its).
        krn_err_h_: OpenCL kernel for computing err_h.
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
        self.krn_err_h_ = None
        self.krn_err_h_clear_ = None

    def initialize(self, **kwargs):
        super(GDPooling, self).initialize(**kwargs)
        self._batch_size = self.h.v.shape[0]
        self._sy = self.h.v.shape[1]
        self._sx = self.h.v.shape[2]
        self._n_channels = self.h.v.size // (self._batch_size * self._sx *
                                             self._sy)
        self._out_sx = self._sx // self.sliding[0] + (
            0 if self._sx % self.sliding[0] == 0 else 1)
        self._out_sy = self._sy // self.sliding[1] + (
            0 if self._sy % self.sliding[1] == 0 else 1)
        output_size = self._n_channels * self._out_sx * self._out_sy * \
            self._batch_size

        if self.err_y.v.size != output_size:
            raise error.ErrBadFormat(
                "Size of err_y differs "
                "from the size computed based on kx, ky, size of h.")

        if self.err_h.v is None or self.err_h.v.size != self.h.v.size:
            self.err_h.reset()
            self.err_h.v = numpy.zeros_like(self.h.v)

        self.err_y.initialize(self.device)
        self.err_h.initialize(self.device)

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
                defines, "%s/gd_pooling_%dx%dx%d_%dx%d.cl" %
                (root.common.cache_dir, self._sx, self._sy, self._n_channels,
                 self.kx, self.ky),
                dtype=self.err_y.v.dtype)

        if self.krn_err_h_clear_ is None:
            self.krn_err_h_clear_ = self.get_kernel("array_clear")
            self.krn_err_h_clear_.set_arg(0, self.err_h.v_)

    def print_times(self, t_start):
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        y = self.err_h.v
        self.debug(
            "%s: %d samples of size %dx%dx%d and sliding %dx%d in %.2f sec" % (
                self.__class__.__name__, y.shape[0], y.shape[2], y.shape[1],
                y.shape[3], self.sliding[0], self.sliding[1],
                time.time() - t_start))

    def ocl_run(self):
        """Do gradient descent.
        """
        self.err_h.unmap()  # we will update err_h
        self.err_y.unmap()  # we will use err_y

        # Clear err_h
        event = self.execute_kernel(self.krn_err_h_clear_,
                                    [self.err_h.v.size], None)
        event.wait()

        # Compute err_h
        event = self.execute_kernel(
            self.krn_err_h_,
            [(self.batch_size or self.err_y.v.shape[0]) *
             (self.err_y.v.size // self.err_y.v.shape[0])], None)
        event.wait()

    def cpu_run(self):
        raise error.ErrNotImplemented()

    def run(self):
        t1 = time.time()
        retval = super(GDPooling, self).run()
        if retval:
            return retval
        self.print_times(t1)


class GDMaxPooling(GDPooling):
    """Gradient Descent for max pooling unit.

    Should be assigned before initialize():
        h_offs

    Updates after run():

    Creates within initialize():

    Attributes:
        h_offs: offsets in err_h where to copy err_y.
        krn_err_h_clear_: OpenCL kernel for setting err_h with zeros.
    """
    def __init__(self, workflow, **kwargs):
        super(GDMaxPooling, self).__init__(workflow, **kwargs)
        self.h_offs = None  # formats.Vector()

    def init_unpickled(self):
        super(GDMaxPooling, self).init_unpickled()
        self.krn_err_h_clear_ = None

    def initialize(self, **kwargs):
        super(GDMaxPooling, self).initialize(**kwargs)

        if self.err_y.v.size != self.h_offs.v.size:
            raise error.ErrBadFormat("Shape of err_y differs from "
                                     "that of h_offs")

        self.h_offs.initialize(self.device)

        if self.device is None:
            return

        if self.krn_err_h_ is None:
            self.krn_err_h_ = self.get_kernel("gd_max_pooling")
            self.krn_err_h_.set_arg(0, self.err_y.v_)
            self.krn_err_h_.set_arg(1, self.err_h.v_)
            self.krn_err_h_.set_arg(2, self.h_offs.v_)

    def ocl_run(self):
        """Do gradient descent on OpenCL device.
        """
        self.h_offs.unmap()  # we will use h_offs
        return super(GDMaxPooling, self).ocl_run()

    def cpu_run(self):
        """Do gradient descent on CPU.
        """
        self.err_y.map_read()
        self.h_offs.map_read()
        self.err_h.map_invalidate()
        self.err_h.v[:] = 0

        if self.kx <= self.sliding[0] and self.ky <= self.sliding[1]:
            # self.h_offs cannot contain equal values - simple assignment
            for err, offset in numpy.nditer([self.err_y.v, self.h_offs]):
                batch, y, x, ch = numpy.unravel_index(offset,
                                                      self.err_h.v.shape)
                self.err_h.v[batch, y, x, ch] = err
        else:
            # self.h_offs can contain equal values
            for err, offset in numpy.nditer([self.err_y.v, self.h_offs]):
                batch, y, x, ch = numpy.unravel_index(offset,
                                                      self.err_h.v.shape)
                self.err_h.v[batch, y, x, ch] += err


class GDAvgPooling(GDPooling):
    """Gradient Descent for avg pooling unit.

    Should be assigned before initialize():

    Updates after run():

    Creates within initialize():

    """
    def initialize(self, **kwargs):
        super(GDAvgPooling, self).initialize(**kwargs)

        if self.device is None:
            return

        if self.krn_err_h_ is None:
            self.krn_err_h_ = self.get_kernel("gd_avg_pooling")
            self.krn_err_h_.set_arg(0, self.err_y.v_)
            self.krn_err_h_.set_arg(1, self.err_h.v_)

    def cpu_run(self):
        self.err_y.map_read()
        self.err_h.map_invalidate()

        if self.kx <= self.sliding[0] and self.ky <= self.sliding[1]:
            # disjoint kernels
            for (batch, y, x, ch), err in numpy.ndenumerate(self.err_y.v):
                hx1 = x * self.sliding[0]
                hx2 = hx1 + self.kx
                hx2 = hx2 if hx2 < self._sx else self._sx
                hy1 = y * self.sliding[1]
                hy2 = hy1 + self.ky
                hy2 = hy2 if hy2 < self._sy else self._sy
                delta = err / float((hx2 - hx1) * (hy2 - hy1))
                for i, j in ((ii, jj) for ii in range(hy1, hy2)
                             for jj in range(hx1, hx2)):
                    self.err_h.v[batch, i, j, ch] = delta
        else:
            # joint kernels
            self.err_h.v[:] = 0
            for (batch, y, x, ch), err in numpy.ndenumerate(self.err_y.v):
                hx1 = x * self.sliding[0]
                hx2 = hx1 + self.kx
                hx2 = hx2 if hx2 < self._sx else self._sx
                hy1 = y * self.sliding[1]
                hy2 = hy1 + self.ky
                hy2 = hy2 if hy2 < self._sy else self._sy
                delta = err / float((hx2 - hx1) * (hy2 - hy1))
                for i, j in ((ii, jj) for ii in range(hy1, hy2)
                             for jj in range(hx1, hx2)):
                    self.err_h.v[batch, i, j, ch] += delta
