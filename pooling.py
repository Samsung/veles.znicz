"""
Created on Dec 3, 2013

Pooling layer.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import formats
import numpy
import pyopencl
import time
import config
import logging
import error


class MaxPooling(units.Forward):
    """Pooling forward propagation.

    Should be assigned before initialize():
        input

    Updates after run():
        output
        input_offs

    Creates within initialize():
        output
        input_offs

    Attributes:
        input: input as batch of multichannel interleaved images.
        output: output as batch of multichannel interleaved images.
        kx: pooling kernel width.
        ky: pooling kernel height.
        krn_: OpenCL kernel.
    """
    def __init__(self, kx=5, ky=5, device=None):
        super(MaxPooling, self).__init__(device=device)
        self.input = None  # formats.Vector(device)
        self.output = formats.Vector(device)
        self.input_offs = formats.Vector(device)
        self.kx = kx
        self.ky = ky
        self.exports.extend(("kx", "ky"))

    def init_unpickled(self):
        super(MaxPooling, self).init_unpickled()
        self.cl_sources_["%s/pooling.cl" % (config.cl_dir)] = ""
        self.krn_ = None

    def initialize(self):
        super(MaxPooling, self).initialize()

        batch_size = self.input.v.shape[0]
        sy = self.input.v.shape[1]
        sx = self.input.v.shape[2]
        n_channels = self.input.v.size // (batch_size * sx * sy)

        out_sx = (sx // self.kx) + (0 if sx % self.kx == 0 else 1)
        out_sy = (sy // self.ky) + (0 if sy % self.ky == 0 else 1)
        output_size = out_sx * out_sy * batch_size
        if self.output.v == None or self.output.v.size != output_size:
            self.output.reset()
            self.output.v = numpy.zeros([batch_size, out_sy, out_sx,
                n_channels], dtype=self.input.v.dtype)

        if (self.input_offs.v == None or
            self.input_offs.v.size != self.output.v.size):
            self.input_offs.reset()
            self.input_offs.v = numpy.zeros(self.output.v.shape,
                                            dtype=numpy.int32)

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.input_offs.initialize(self.device)

        if self.device == None:
            return

        if self.krn_ == None:
            defines = ("%s\n"
                       "#define SX %d\n"
                       "#define SY %d\n"
                       "#define N_CHANNELS %d\n"
                       "#define KX %d\n"
                       "#define KY %d\n"
                       "\n" % (
                       config.cl_defines[config.c_dtype],
                       sx, sy, n_channels, self.kx, self.ky))
            self.build_program(defines,
                "%s/max_pooling_%dx%dx%d_%dx%d.cl" % (
                config.cache_dir, sx, sy, n_channels, self.kx, self.ky))

            self.krn_ = pyopencl.Kernel(self.prg_, "do_max_pooling")
            self.krn_.set_arg(0, self.input.v_)
            self.krn_.set_arg(1, self.output.v_)
            self.krn_.set_arg(2, self.input_offs.v_)

    def print_times(self, t_start):
        """Show some statistics.
        """
        log = self.log()
        if not log.isEnabledFor(logging.DEBUG):
            return
        y = self.input.v
        self.log().debug("%s: %d samples of size %dx%dx%d vs "
                         "pooling window of size %dx%d in %.2f sec" % (
            self.__class__.__name__, y.shape[0], y.shape[2], y.shape[1],
            y.shape[3], self.kx, self.ky, time.time() - t_start))

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        self.input_offs.unmap()  # we will be updating input_offs
        self.output.unmap()  # we will be updating output
        self.input.unmap()  # we will use input
        y = self.output.v
        global_size = [y.shape[2], y.shape[0] * y.shape[1]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_,
                                                 global_size, None)
        event.wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        raise error.ErrNotImplemented()

    def run(self):
        t1 = time.time()
        retval = super(MaxPooling, self).run()
        if retval:
            return retval
        self.print_times(t1)
