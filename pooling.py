"""
==============
Pooling layers
==============

A short description of `pooling` (aka `subsampling`) can be found `here \
<http://white.stanford.edu/teach/index.php/\
An_Introduction_to_Convolutional_Neural_Networks#Subsampling>`_.

Pooling types implemented:

- `AvgPooling`: averaging pooling
- `MaxPooling`: maximum selection pooling
- `StochasticPooling`: stochastic pooling, described in article `"Stochastic \
    Pooling for Regularization of Deep Convolutional Neural Networks" \
    <http://www.matthewzeiler.com/pubs/iclr2013/iclr2013.pdf>`_.


Created on Dec 3, 2013.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import logging
import numpy
import time
from zope.interface import implementer

import veles.error as error
import veles.formats as formats
from veles.opencl_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units
from veles.distributable import IDistributable, TriviallyDistributable
from veles.tests import DummyWorkflow
import veles.prng as prng


@implementer(IOpenCLUnit, IDistributable)
class Pooling(TriviallyDistributable, nn_units.Forward):
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
        try:
            kx = kwargs["kx"]
            ky = kwargs["ky"]
        except KeyError:
            raise KeyError("kx and ky are required constructor parameters")
        sliding = kwargs.get("sliding", (kx, ky))
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        kwargs["sliding"] = sliding
        super(Pooling, self).__init__(workflow, **kwargs)
        self.kx = kx
        self.ky = ky
        self.sliding = sliding
        self.exports.extend(("kx", "ky", "sliding"))
        self._no_output = False

    def init_unpickled(self):
        super(Pooling, self).init_unpickled()
        self.cl_sources_["pooling.cl"] = {}
        if not hasattr(self, "_no_output"):
            self._no_output = False
        if not hasattr(self, "uniform"):
            self.uniform = None

    def create_output(self):
        self._batch_size = self.input.mem.shape[0]
        self._sy = self.input.mem.shape[1]
        self._sx = self.input.mem.shape[2]
        self._n_channels = self.input.mem.size // (self._batch_size *
                                                   self._sx * self._sy)

        last_x = self._sx - self.kx
        last_y = self._sy - self.ky
        if last_x % self.sliding[0] == 0:
            self._out_sx = last_x // self.sliding[0] + 1
        else:
            self._out_sx = last_x // self.sliding[0] + 2
        if last_y % self.sliding[1] == 0:
            self._out_sy = last_y // self.sliding[1] + 1
        else:
            self._out_sy = last_y // self.sliding[1] + 2

        self._output_size = self._n_channels * self._out_sx * self._out_sy * \
            self._batch_size
        self._output_shape = [self._batch_size, self._out_sy, self._out_sx,
                              self._n_channels]
        if (not self._no_output and (not self.output or
                                     self.output.size != self._output_size)):
            self.output.reset()
            self.output.mem = numpy.zeros(self._output_shape,
                                          dtype=self.input.mem.dtype)

    def initialize(self, device, **kwargs):
        super(Pooling, self).initialize(device=device, **kwargs)

        self.create_output()

        self.input.initialize(self.device)
        if not self._no_output:
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
        sh = self._output_shape
        global_size = [sh[3] * sh[2], sh[1] * sh[0]]
        self.execute_kernel(global_size, None)

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
                val = self.cpu_run_cut(cut, (batch, y1, x1, ch, out_y, out_x))
                self.output.mem[batch, out_y, out_x, ch] = val

    def run(self):
        t1 = time.time()
        retval = super(Pooling, self).run()
        if retval:
            return retval
        self.print_debug_data(t1)


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

    def initialize(self, device, **kwargs):
        super(OffsetPooling, self).initialize(device=device, **kwargs)

        if (not self._no_output and (
                not self.input_offset or
                self.input_offset.size != self.output.size)):
            self.input_offset.reset()
            self.input_offset.mem = numpy.zeros(self.output.mem.shape,
                                                dtype=numpy.int32)

        if not self._no_output:
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
        batch, y1, x1, ch, out_y, out_x = coords
        cut_index = self.cpu_run_cut_offset(
            cut, numpy.ravel_multi_index((batch, out_y, out_x, ch),
                                         self.output.mem.shape))
        i, j = numpy.unravel_index(cut_index, cut.shape)
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

    def apply_data_from_master(self, data):
        self.input_offset.map_invalidate()
        self.input_offset.mem[:] = data[0][:]


class MaxPoolingBase(OffsetPooling):
    """MaxPooling forward propagation base class.
    """

    def initialize(self, device, **kwargs):
        super(MaxPoolingBase, self).initialize(device=device, **kwargs)

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

    Attributes:
        uniform: instance of veles.prng.Uniform.
    """
    def __init__(self, workflow, **kwargs):
        super(StochasticPoolingBase, self).__init__(workflow, **kwargs)
        self.uniform = kwargs.get("uniform")

    def init_unpickled(self):
        super(StochasticPoolingBase, self).init_unpickled()
        self._rand_set = False
        self._rand_arg = 3
        self._kernel_name = "do_stochastic_pooling"

    def initialize(self, device, **kwargs):
        super(StochasticPoolingBase, self).initialize(device=device, **kwargs)

        if self.uniform is None:
            self.uniform = prng.Uniform(DummyWorkflow())

        if self.uniform.output_bytes < (self._output_size << 1):
            if self.uniform.is_initialized:
                raise error.AlreadyExistsError(
                    "uniform is already initialized and "
                    "has not enough output size")
            self.uniform.output_bytes = self._output_size << 1

        if self.device is None:
            return

        self.assign_kernel(self._kernel_name)
        self.set_args()

    def cpu_run(self):
        if not self.uniform.is_initialized:
            self.uniform.initialize(self.device)
            self.info("Initialized StochasticPoolingBase.uniform with "
                      "output_bytes=%d", self.uniform.output_bytes)
        self.uniform.fill_cpu(self._output_size << 1)
        super(StochasticPoolingBase, self).cpu_run()

    def ocl_run(self):
        if not self.uniform.is_initialized:
            self.uniform.initialize(self.device)
            self.info("Initialized StochasticPoolingBase.uniform with "
                      "output_bytes=%d", self.uniform.output_bytes)
        if not self._rand_set:
            self.set_arg(self._rand_arg, self.uniform.output)
            self._rand_set = True
        self.uniform.fill_ocl(self._output_size << 1)
        super(StochasticPoolingBase, self).ocl_run()

    def calculate_position_cpu(self, index, vsum):
        rnd = self.uniform.output.mem.view(dtype=numpy.uint16)[index]
        return rnd * vsum / 65536

    def calculate_random_index_cpu(self, cut, index):
        rnd = self.uniform.output.mem.view(dtype=numpy.uint16)[index]
        return int(rnd * cut.size >> 16)


class StochasticPooling(StochasticPoolingBase):
    """StochasticPooling forward propagation.
    """

    def cpu_run_cut_offset(self, cut, index):
        vsum = numpy.sum(cut[cut > 0])
        if vsum == 0:
            return self.calculate_random_index_cpu(cut, index)
        position = self.calculate_position_cpu(index, vsum)
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
        vsum = numpy.sum(numpy.abs(cut))
        if vsum == 0:
            return self.calculate_random_index_cpu(cut, index)
        position = self.calculate_position_cpu(index, vsum)
        vsum = 0
        for i in range(cut.size):
            val = cut.ravel()[i]
            vsum += abs(val)
            if position <= vsum:
                return i


class StochasticPoolingDepooling(StochasticPooling):
    """Stochastic pooling with depooling in-place.
    """
    def __init__(self, workflow, **kwargs):
        super(StochasticPoolingDepooling, self).__init__(workflow, **kwargs)
        self._no_output = True

    def init_unpickled(self):
        super(StochasticPoolingDepooling, self).init_unpickled()
        self.cl_sources_["pooling.cl"]["USE_POOLING_DEPOOLING"] = 1
        self._rand_arg = 1
        self._kernel_name = "do_stochastic_pooling_depooling"

    def set_args(self, *args):
        self.set_arg(0, self.input)

    def cpu_run(self):
        raise RuntimeError("Not implemented")


class StochasticAbsPoolingDepooling(StochasticPoolingDepooling):
    """Stochastic abs pooling with depooling in-place.
    """
    def __init__(self, workflow, **kwargs):
        super(StochasticAbsPoolingDepooling, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(StochasticAbsPoolingDepooling, self).init_unpickled()
        self.cl_sources_["pooling.cl"]["ABS_VALUES"] = 1


class AvgPooling(Pooling):
    """AvgPooling forward propagation.

    Must be assigned before initialize():

    Updates after run():

    Creates within initialize():

    """
    def initialize(self, device, **kwargs):
        super(AvgPooling, self).initialize(device=device, **kwargs)

        if self.device is None:
            return

        self.assign_kernel("do_avg_pooling")
        self.set_args(self.input, self.output)

    def cpu_run_cut(self, cut, coords):
        return numpy.sum(cut) / cut.size
