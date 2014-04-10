"""
Created on Dec 3, 2013

Gradient Descent for Pooling units.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import logging
import numpy
import time

from veles.config import root
import veles.error as error
import veles.znicz.nn_units as nn_units


class GDPooling(nn_units.GD):
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

    def initialize(self):
        super(GDPooling, self).initialize()
        batch_size = self.h.v.shape[0]
        sy = self.h.v.shape[1]
        sx = self.h.v.shape[2]
        n_channels = self.h.v.size // (batch_size * sx * sy)

        out_sx = sx // self.sliding[0] + (
            0 if sx % self.sliding[0] == 0 else 1)
        out_sy = sy // self.sliding[1] + (
            0 if sy % self.sliding[1] == 0 else 1)
        output_size = n_channels * out_sx * out_sy * batch_size

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

        if self.prg_ is None:
            defines = {
                'SX': sx,
                'SY': sy,
                'N_CHANNELS': n_channels,
                'KX': self.kx,
                'KY': self.ky,
                'SLIDE_X': self.sliding[0],
                'SLIDE_Y': self.sliding[1]
            }
            self.build_program(
                defines, "%s/gd_pooling_%dx%dx%d_%dx%d.cl" %
                (root.common.cache_dir, sx, sy, n_channels, self.kx, self.ky),
                dtype=self.err_y.v.dtype)

    def print_times(self, t_start):
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        y = self.err_h.v
        self.debug(
            "%s: %d samples of size %dx%dx%d and sliding %dx%d in %.2f sec" % (
                self.__class__.__name__, y.shape[0], y.shape[2], y.shape[1],
                y.shape[3], self.sliding[0], self.sliding[1],
                time.time() - t_start))

    def gpu_run(self):
        """Do gradient descent.
        """
        self.err_h.unmap()  # we will update err_h
        self.err_y.unmap()  # we will use err_y
        event = self.execute_kernel(self.krn_err_h_,
                                    [self.err_y.v.size], None)
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

    def initialize(self):
        super(GDMaxPooling, self).initialize()

        if self.err_y.v.size != self.h_offs.v.size:
            raise error.ErrBadFormat("Shape of err_y differs from "
                                     "that of h_offs")

        self.h_offs.initialize(self.device)

        if self.device is None:
            return

        if self.krn_err_h_clear_ is None:
            self.krn_err_h_clear_ = self.get_kernel("array_clear")
            self.krn_err_h_clear_.set_arg(0, self.err_h.v_)

        if self.krn_err_h_ is None:
            self.krn_err_h_ = self.get_kernel("gd_max_pooling")
            self.krn_err_h_.set_arg(0, self.err_y.v_)
            self.krn_err_h_.set_arg(1, self.err_h.v_)
            self.krn_err_h_.set_arg(2, self.h_offs.v_)

    def gpu_run(self):
        """Do gradient descent.
        """
        self.err_h.unmap()  # we will clear err_h here
        event = self.execute_kernel(self.krn_err_h_clear_,
                                    [self.err_h.v.size], None)
        event.wait()
        self.h_offs.unmap()  # we will use h_offs
        return super(GDMaxPooling, self).gpu_run()


class GDAvgPooling(GDPooling):
    """Gradient Descent for avg pooling unit.

    Should be assigned before initialize():

    Updates after run():

    Creates within initialize():

    """
    def init_unpickled(self):
        super(GDAvgPooling, self).init_unpickled()
        self.krn_err_h_clear_ = None

    def initialize(self):
        super(GDAvgPooling, self).initialize()

        if self.device is None:
            return

        if (self.krn_err_h_clear_ is None and
           (self.sliding[0] < self.kx or self.sliding[1] < self.ky)):
            self.krn_err_h_clear_ = self.get_kernel("array_clear")
            self.krn_err_h_clear_.set_arg(0, self.err_h.v_)

        if self.krn_err_h_ is None:
            self.krn_err_h_ = self.get_kernel("gd_avg_pooling")
            self.krn_err_h_.set_arg(0, self.err_y.v_)
            self.krn_err_h_.set_arg(1, self.err_h.v_)

    def gpu_run(self):
        """Do gradient descent.
        """
        if self.krn_err_h_clear_ is not None:
            self.err_h.unmap()  # we will clear err_h here
            event = self.execute_kernel(self.krn_err_h_clear_,
                                        [self.err_h.v.size], None)
            event.wait()
        return super(GDAvgPooling, self).gpu_run()
