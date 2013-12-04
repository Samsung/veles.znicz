"""
Created on Dec 3, 2013

Gradient Descent for Pooling units.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import time
import pyopencl
import config
import logging
import formats
import numpy
import error


class GDMaxPooling(units.GD):
    """Gradient Descent for max pooling unit.

    Should be assigned before initialize():
        err_y
        h
        h_offs

    Updates after run():
        err_h

    Creates within initialize():
        err_h

    Attributes:
        err_y: backpropagation errors for y.
        h: input (will get only shape from it).
        err_h: backpropagation errors for h (will compute its).
        h_offs: offsets in err_h where to copy err_y.
        krn_err_h_clear_: OpenCL kernel for setting err_h with zeros.
        krn_err_h_: OpenCL kernel for computing err_h.
    """
    def __init__(self, device=None):
        super(GDMaxPooling, self).__init__(device=device)
        self.err_y = None  # formats.Vector()
        self.h = None  # formats.Vector()
        self.h_offs = None  # formats.Vector()
        self.err_h = formats.Vector()

    def init_unpickled(self):
        super(GDMaxPooling, self).init_unpickled()
        self.cl_sources_["%s/gradient_descent_pooling.cl" % (
                                                config.cl_dir)] = ""
        self.krn_err_h_clear_ = None
        self.krn_err_h_ = None

    def initialize(self):
        if self.err_y.v.size != self.h_offs.v.size:
            raise error.ErrBadFormat("Shape of err_y differs from "
                                     "one of h_offs")

        if self.err_h.v == None or self.err_h.v.size != self.h.v.size:
            self.err_h.reset()
            self.err_h.v = numpy.zeros_like(self.h.v)

        self.err_y.initialize(self.device)
        self.err_h.initialize(self.device)
        self.h_offs.initialize(self.device)

        if self.device == None:
            return

        if self.prg_ == None:
            defines = ("%s\n" % (
                       config.cl_defines[config.c_dtype]))
            self.build_program(defines, "%s/gd_pooling_%d_%d.cl" % (
                config.cache_dir,
                self.err_h.v.size // self.err_h.v.shape[0],
                self.err_y.v.size // self.err_y.v.shape[0]))

            self.krn_err_h_clear_ = pyopencl.Kernel(self.prg_, "array_clear")
            self.krn_err_h_clear_.set_arg(0, self.err_h.v_)

            self.krn_err_h_ = pyopencl.Kernel(self.prg_, "gd_max_pooling")
            self.krn_err_h_.set_arg(0, self.err_y.v_)
            self.krn_err_h_.set_arg(1, self.err_h.v_)
            self.krn_err_h_.set_arg(2, self.h_offs.v_)

    def print_times(self, t_start):
        log = self.log()
        if not log.isEnabledFor(logging.DEBUG):
            return
        y = self.err_h.v
        self.log().debug("%s: %d samples of size %dx%dx%d in %.2f sec" % (
            self.__class__.__name__, y.shape[0], y.shape[2], y.shape[1],
            y.shape[3], time.time() - t_start))

    def gpu_run(self):
        """Do gradient descent.
        """
        self.err_h.unmap()  # we will update err_h
        self.err_y.unmap()  # we will use err_y
        self.h_offs.unmap()  # we will use h_offs
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
            self.krn_err_h_clear_, [self.err_h.v.size], None)
        event.wait()
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
            self.krn_err_h_, [self.err_y.v.size], None)
        event.wait()

    def cpu_run(self):
        raise error.ErrNotImplemented()

    def run(self):
        t1 = time.time()
        retval = super(GDMaxPooling, self).run()
        if retval:
            return retval
        self.print_times(t1)
