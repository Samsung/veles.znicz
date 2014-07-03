"""
Created on Dec 3, 2013

Pooling layer.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import logging
import numpy
import time
from zope.interface import implementer

import veles.formats as formats
from veles.opencl_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units
from veles.distributable import IDistributable
import veles.random_generator as random_generator


@implementer(IOpenCLUnit)
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
        self._n_channels = self.input.mem.size // (self._batch_size *
                                                   self._sx * self._sy)
        self._out_sx = self._sx // self.sliding[0] + (
            0 if self._sx % self.sliding[0] == 0 else 1)
        self._out_sy = self._sy // self.sliding[1] + (
            0 if self._sy % self.sliding[1] == 0 else 1)
        self._output_size = self._n_channels * self._out_sx * self._out_sy * \
            self._batch_size
        if (self.output.mem is None or
                self.output.mem.size != self._output_size):
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
        if not self.logger.isEnabledFor(logging.DEBUG):
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
        self.execute_kernel(global_size, None)

    def cpu_run(self):
        self.input.map_read()
        self.output.map_invalidate()
        cut_index = 0
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
                val = self.cpu_run_cut(cut, (cut_index, batch, y1, x1, ch,
                                       out_y, out_x))
                cut_index += 1
                self.output.mem[batch, out_y, out_x, ch] = val

    def run(self):
        t1 = time.time()
        retval = super(Pooling, self).run()
        if retval:
            return retval
        self.print_debug_data(t1)

    # IDistributable implementation
    def generate_data_for_slave(self, slave):
        return None

    def generate_data_for_master(self):
        return None

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        pass

    def drop_slave(self, slave):
        pass


@implementer(IDistributable)
class OffsetPooling(Pooling):
    """Pooling by offset forward propagation.

    Must be assigned before initialize():

    Updates after run():
        input_offset

    Creates within initialize():
        input_offset

    Attributes:
        input_offset: offsets in the input where elements are passed through.
    """
    def __init__(self, workflow, **kwargs):
        super(OffsetPooling, self).__init__(workflow, **kwargs)
        self.input_offset = formats.Vector()
        self.demand("input")

    def initialize(self, **kwargs):
        super(OffsetPooling, self).initialize(**kwargs)

        if (self.input_offset.mem is None or
                self.input_offset.mem.size != self.output.mem.size):
            self.input_offset.reset()
            self.input_offset.mem = numpy.zeros(self.output.mem.shape,
                                                dtype=numpy.int32)

        self.input_offset.initialize(self.device)

    def set_args(self, *args):
        super(OffsetPooling, self).set_args(self.input, self.output,
                                            self.input_offset, *args)

    def ocl_run(self):
        self.input_offset.unmap()  # we will be updating input_offset
        super(OffsetPooling, self).ocl_run()

    def cpu_run(self):
        self.input_offset.map_invalidate()
        super(OffsetPooling, self).cpu_run()

    def cpu_run_cut(self, cut, coords):
        index, batch, y1, x1, ch, out_y, out_x = coords
        cut_inner_index = self.cpu_run_cut_offset(cut, index)
        i, j = numpy.unravel_index(cut_inner_index, cut.shape)
        idx = numpy.ravel_multi_index((batch, y1 + i, x1 + j, ch),
                                      self.input.mem.shape)
        val = numpy.ravel(self.input.mem)[idx]
        self.input_offset.mem[batch, out_y, out_x, ch] = idx
        return val

    # IDistributable implementation
    def generate_data_for_slave(self, slave):
        self.input_offset.map_read()
        data = (self.input_offset.mem)
        return data

    def generate_data_for_master(self):
        return None

    def apply_data_from_slave(self, data, slave):
        pass

    def apply_data_from_master(self, data):
        self.input_offset.map_invalidate()
        self.input_offset.mem[:] = data[0][:]

    def drop_slave(self, slave):
        pass


class MaxPoolingBase(OffsetPooling):
    """MaxPooling forward propagation base class.
    """

    def initialize(self, **kwargs):
        super(MaxPoolingBase, self).initialize(**kwargs)

        if self.device is None:
            return

        self.assign_kernel("do_max_pooling")
        self.set_args()


class MaxPooling(MaxPoolingBase):
    """MaxPooling forward propagation.
    """

    def cpu_run_cut_offset(self, cut, index):
        return cut.argmax()


class MaxAbsPooling(MaxPoolingBase):
    """MaxAbsPooling forward propagation.

    Must be assigned before initialize():

    Updates after run():
        input_offset

    Creates within initialize():
        input_offset

    Attributes:
        input_offset: offsets in the input where maximum elements were found.
    """

    def __init__(self, workflow, **kwargs):
        super(MaxAbsPooling, self).__init__(workflow, **kwargs)
        self.cl_sources_["pooling.cl"] = {"ABS_VALUES": 1}

    def cpu_run_cut_offset(self, cut, index):
        return numpy.abs(cut).argmax()


class StochasticPoolingBase(OffsetPooling):
    """Stochastic pooling forward propagation.

    Must be assigned before initialize():

    Updates after run():
        input_offset

    Creates within initialize():
        input_offset
        _random_states

    Attributes:
        input_offset: random offset in the input, probability of which is
        proportional to input values. In case of negative inputs, treat them
        as zeros.
    """

    MIN_RANDOM_STATE = 0
    MAX_RANDOM_STATE = 0x100000000

    def __init__(self, workflow, **kwargs):
        super(StochasticPoolingBase, self).__init__(workflow, **kwargs)
        self._random_states = formats.Vector()
        self.rand = random_generator.get()

    def initialize(self, **kwargs):
        super(StochasticPoolingBase, self).initialize(**kwargs)

        self._random_states.mem = self.rand.randint(
            low=StochasticPoolingBase.MIN_RANDOM_STATE,
            high=StochasticPoolingBase.MAX_RANDOM_STATE,
            size=self.output.mem.size * 4).astype(numpy.uint32)

        if self.device is None:
            return

        self._random_states.initialize(self.device)

        self.assign_kernel("do_stochastic_pooling")
        self.set_args(self._random_states)


class StochasticPooling(StochasticPoolingBase):
    """StochasticPooling forward propagation.
    """

    def cpu_run_cut_offset(self, cut, index):
        vsum = numpy.sum(cut[cut > 0])
        position = random_generator.RandomGenerator.xorshift128plus(
            self._random_states, index * 2) / ((1 << 64) - 1.) * vsum
        vsum = 0
        for i in range(cut.size):
            val = cut.ravel()[i]
            if val > 0:
                vsum += val
            if position <= vsum:
                return i


class StochasticAbsPooling(StochasticPoolingBase):
    """StochasticAbsPooling forward propagation.
    """

    def __init__(self, workflow, **kwargs):
        super(StochasticAbsPooling, self).__init__(workflow, **kwargs)
        self.cl_sources_["pooling.cl"] = {"ABS_VALUES": 1}

    def cpu_run_cut_offset(self, cut, index):
        vsum = numpy.sum(cut[cut > 0])
        position = random_generator.RandomGenerator.xorshift128plus(
            self._random_states, index * 2) / ((1 << 64) - 1.) * vsum
        vsum = 0
        for i in range(cut.size):
            val = cut.ravel()[i]
            vsum += abs(val)
            if position <= vsum:
                return i


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

    def cpu_run_cut(self, cut, coords):
        return numpy.sum(cut) / cut.size
