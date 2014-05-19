"""
Created on Dec 3, 2013

Pooling layer.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import time

import veles.error as error
import veles.formats as formats
import veles.znicz.nn_units as nn_units


class Pooling(nn_units.Forward):
    """Pooling forward propagation.

    Must be assigned before initialize():
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

    def initialize(self, **kwargs):
        super(Pooling, self).initialize(**kwargs)

        self._batch_size = self.input.mem.shape[0]
        self._sy = self.input.mem.shape[1]
        self._sx = self.input.mem.shape[2]
        self._n_channels = self.input.mem.size // (self._batch_size * self._sx *
                                                 self._sy)
        self._out_sx = self._sx // self.sliding[0] + (
            0 if self._sx % self.sliding[0] == 0 else 1)
        self._out_sy = self._sy // self.sliding[1] + (
            0 if self._sy % self.sliding[1] == 0 else 1)
        self._output_size = self._n_channels * self._out_sx * self._out_sy * \
            self._batch_size
        if self.output.mem is None or self.output.mem.size != self._output_size:
            self.output.reset()
            self.output.mem = numpy.zeros(
                [self._batch_size, self._out_sy, self._out_sx,
                 self._n_channels],
                dtype=self.input.mem.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)

        if self.device is None:
            return

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
            defines, "pooling_%dx%dx%d_%dx%d.cl" %
            (self._sx, self._sy, self._n_channels,
             self.kx, self.ky), dtype=self.input.mem.dtype)

    def print_debug_data(self, t_start):
        """Show some statistics.
        """
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        y = self.input.mem
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
        y = self.output.mem
        global_size = [y.shape[3] * y.shape[2], y.shape[1] * y.shape[0]]
        self.execute_kernel(global_size, None).wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        raise error.ErrNotImplemented()

    def run(self):
        t1 = time.time()
        retval = super(Pooling, self).run()
        if retval:
            return retval
        self.print_debug_data(t1)


class MaxPooling(Pooling):
    """MaxPooling forward propagation.

    Must be assigned before initialize():

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

    def initialize(self, **kwargs):
        super(MaxPooling, self).initialize(**kwargs)

        if (self.input_offs.mem is None or
                self.input_offs.mem.size != self.output.mem.size):
            self.input_offs.reset()
            self.input_offs.mem = numpy.zeros(self.output.mem.shape,
                                            dtype=numpy.int32)

        self.input_offs.initialize(self.device)

        if self.device is None:
            return

        self.assign_kernel("do_max_pooling")
        self.set_args(self.input, self.output, self.input_offs)

    def ocl_run(self):
        self.input_offs.unmap()  # we will be updating input_offs
        return super(MaxPooling, self).ocl_run()

    def cpu_run(self):
        self.input.map_read()
        abs_input = numpy.fabs(self.input.mem)
        self.input_offs.map_invalidate()
        self.output.map_invalidate()
        for batch, ch in ((batch, ch) for batch in range(self._batch_size)
                          for ch in range(self._n_channels)):
            for out_x, out_y in ((out_x, out_y)
                                 for out_x in range(self._out_sx)
                                 for out_y in range(self._out_sy)):
                x1 = out_x * self.sliding[0]
                y1 = out_y * self.sliding[1]
                test_idx = x1 + self.kx
                x2 = test_idx if test_idx <= self._sx else self._sx
                test_idx = y1 + self.ky
                y2 = test_idx if test_idx <= self._sy else self._sy
                cut = abs_input[batch, y1:y2, x1:x2, ch]
                i, j = numpy.unravel_index(cut.argmax(), cut.shape)
                idx = numpy.ravel_multi_index((batch, y1 + i, x1 + j, ch),
                                              self.input.mem.shape)
                val = numpy.ravel(self.input.mem)[idx]
                self.input_offs.mem[batch, out_y, out_x, ch] = idx
                self.output.mem[batch, out_y, out_x, ch] = val


class AvgPooling(Pooling):
    """AvgPooling forward propagation.

    Must be assigned before initialize():

    Updates after run():

    Creates within initialize():

    """
    def initialize(self, **kwargs):
        super(AvgPooling, self).initialize(**kwargs)

        if self.device is None:
            return

        self.assign_kernel("do_avg_pooling")
        self.set_args(self.input, self.output)

    def cpu_run(self):
        self.input.map_read()
        self.output.map_invalidate()
        for batch, ch in ((batch, ch) for batch in range(self._batch_size)
                          for ch in range(self._n_channels)):
            for out_x, out_y in ((out_x, out_y)
                                 for out_x in range(self._out_sx)
                                 for out_y in range(self._out_sy)):
                x1 = out_x * self.sliding[0]
                y1 = out_y * self.sliding[1]
                test_idx = x1 + self.kx
                x2 = test_idx if test_idx <= self._sx else self._sx
                test_idx = y1 + self.ky
                y2 = test_idx if test_idx <= self._sy else self._sy
                cut = self.input.mem[batch, y1:y2, x1:x2, ch]
                val = numpy.sum(cut) / cut.size
                self.output.mem[batch, out_y, out_x, ch] = val
