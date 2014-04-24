"""
Created on Dec 3, 2013

Pooling layer.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import time

from veles.config import root
import veles.error as error
import veles.formats as formats
import veles.znicz.nn_units as nn_units


class Pooling(nn_units.Forward):
    """Pooling forward propagation.

    Should be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: input as batch of multichannel interleaved images.
        output: output as batch of multichannel interleaved images.
        kx: pooling kernel width.
        ky: pooling kernel height.
        sliding: tuple of kernel sliding (by x-axis, by y-axis).
        krn_: OpenCL kernel.
    """
    def __init__(self, workflow, **kwargs):
        kx = kwargs.get("kx", 2)
        ky = kwargs.get("ky", 2)
        sliding = kwargs.get("sliding", (kx, ky))
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        kwargs["sliding"] = sliding
        super(Pooling, self).__init__(workflow, **kwargs)
        self.kx = kx
        self.ky = ky
        self.sliding = sliding
        self.exports.extend(("kx", "ky", "sliding"))

    def init_unpickled(self):
        super(Pooling, self).init_unpickled()
        self.cl_sources_["pooling.cl"] = {}
        self.krn_ = None

    def initialize(self):
        super(Pooling, self).initialize()

        batch_size = self.input.v.shape[0]
        sy = self.input.v.shape[1]
        sx = self.input.v.shape[2]
        n_channels = self.input.v.size // (batch_size * sx * sy)

        out_sx = sx // self.sliding[0] + (
            0 if sx % self.sliding[0] == 0 else 1)
        out_sy = sy // self.sliding[1] + (
            0 if sy % self.sliding[1] == 0 else 1)
        output_size = n_channels * out_sx * out_sy * batch_size
        if self.output.v is None or self.output.v.size != output_size:
            self.output.reset()
            self.output.v = numpy.zeros(
                [batch_size, out_sy, out_sx, n_channels],
                dtype=self.input.v.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)

        if self.device is None:
            return

        if self.krn_ is None:
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
                defines, "%s/pooling_%dx%dx%d_%dx%d.cl" %
                (root.common.cache_dir, sx, sy, n_channels, self.kx, self.ky),
                dtype=self.input.v.dtype)

    def print_times(self, t_start):
        """Show some statistics.
        """
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        y = self.input.v
        self.debug(
            "%s: %d samples of size %dx%dx%d vs "
            "pooling window of size %dx%d and sliding %dx%d in %.2f sec" %
            (self.__class__.__name__, y.shape[0], y.shape[2], y.shape[1],
             y.shape[3], self.kx, self.ky, self.sliding[0], self.sliding[1],
             time.time() - t_start))

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()  # we will be updating output
        self.input.unmap()  # we will use input
        y = self.output.v
        global_size = [y.shape[3] * y.shape[2], y.shape[1] * y.shape[0]]
        event = self.execute_kernel(self.krn_, global_size, None)
        event.wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        raise error.ErrNotImplemented()

    def run(self):
        t1 = time.time()
        retval = super(Pooling, self).run()
        if retval:
            return retval
        self.print_times(t1)


class MaxPooling(Pooling):
    """MaxPooling forward propagation.

    Should be assigned before initialize():

    Updates after run():
        input_offs

    Creates within initialize():
        input_offs

    Attributes:
        input_offs: offsets in the input where maximum elements were found.
    """
    def __init__(self, workflow, **kwargs):
        super(MaxPooling, self).__init__(workflow, **kwargs)
        self.input_offs = formats.Vector()

    def initialize(self):
        super(MaxPooling, self).initialize()

        if (self.input_offs.v is None or
                self.input_offs.v.size != self.output.v.size):
            self.input_offs.reset()
            self.input_offs.v = numpy.zeros(self.output.v.shape,
                                            dtype=numpy.int32)

        self.input_offs.initialize(self.device)

        if self.device is None:
            return

        if self.krn_ is None:
            self.krn_ = self.get_kernel("do_max_pooling")
            self.krn_.set_arg(0, self.input.v_)
            self.krn_.set_arg(1, self.output.v_)
            self.krn_.set_arg(2, self.input_offs.v_)

    def ocl_run(self):
        self.input_offs.unmap()  # we will be updating input_offs
        return super(MaxPooling, self).ocl_run()


class AvgPooling(Pooling):
    """AvgPooling forward propagation.

    Should be assigned before initialize():

    Updates after run():

    Creates within initialize():

    """
    def initialize(self):
        super(AvgPooling, self).initialize()

        if self.device is None:
            return

        if self.krn_ is None:
            self.krn_ = self.get_kernel("do_avg_pooling")
            self.krn_.set_arg(0, self.input.v_)
            self.krn_.set_arg(1, self.output.v_)
