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
from itertools import product
import logging
import numpy
import time
from zope.interface import implementer

import veles.error as error
import veles.memory as formats
from veles.accelerated_units import IOpenCLUnit, ICUDAUnit
import veles.znicz.nn_units as nn_units
from veles.distributable import IDistributable, TriviallyDistributable
from veles.prng.uniform import Uniform
from veles.units import Unit


class PoolingBase(Unit):
    POOL_ATTRS = ("kx", "ky", "sliding")

    def __init__(self, workflow, **kwargs):
        super(PoolingBase, self).__init__(workflow, **kwargs)
        self._output_shape = tuple()
        self._out_sxy = tuple()

    @property
    def output_shape(self):
        if self._output_shape == tuple():
            self._output_shape = (self.input_batch_size, self.out_sy,
                                  self.out_sx, self.n_channels)
        return self._output_shape

    @property
    def output_size(self):
        return int(numpy.prod(self.output_shape))

    @property
    def input_batch_size(self):
        return self.input.shape[0]

    @property
    def sy(self):
        return self.input.shape[1]

    @property
    def sx(self):
        return self.input.shape[2]

    @property
    def out_sxy(self):
        if self._out_sxy == tuple():
            outs = [0, 0]
            for i, last in enumerate((self.sx - self.kx, self.sy - self.ky)):
                outs[i] = last // self.sliding[i] + 1
                if last % self.sliding[i] != 0:
                    outs[i] += 1

            self._out_sxy = tuple(outs)
        return self._out_sxy

    @property
    def out_sx(self):
        return self.out_sxy[0]

    @property
    def out_sy(self):
        return self.out_sxy[1]

    @property
    def n_channels(self):
        return self.input.size // (self.input_batch_size * self.sx * self.sy)


@implementer(IOpenCLUnit, ICUDAUnit, IDistributable)
class Pooling(PoolingBase, nn_units.Forward, TriviallyDistributable):
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
    MAPPING = set()

    def __init__(self, workflow, **kwargs):
        super(Pooling, self).__init__(workflow, **kwargs)
        self.kx = kwargs["kx"]
        self.ky = kwargs["ky"]
        self.sliding = kwargs.get("sliding") or (self.kx, self.ky)
        self.exports.extend(self.POOL_ATTRS)
        self._no_output = False

    def init_unpickled(self):
        super(Pooling, self).init_unpickled()
        self.sources_["pooling"] = {}
        if not hasattr(self, "_no_output"):
            self._no_output = False
        if not hasattr(self, "uniform"):
            self.uniform = None

    def initialize(self, device, **kwargs):
        super(Pooling, self).initialize(device=device, **kwargs)

        if not self._no_output:
            if not self.output:
                self.output.reset(numpy.zeros(self.output_shape,
                                              dtype=self.input.dtype))
            else:
                assert self.output.shape == self.output_shape, \
                    "%s != %s" % (self.output.shape, self.output_shape)
            self.output.initialize(self.device)

        self.input.initialize(self.device)

    def _gpu_init(self):
        defines = {
            'SX': self.sx,
            'SY': self.sy,
            'N_CHANNELS': self.n_channels,
            'KX': self.kx,
            'KY': self.ky,
            'SLIDE_X': self.sliding[0],
            'SLIDE_Y': self.sliding[1],
            'OUTPUT_SIZE': self.output_size
        }
        self.build_program(
            defines, "%s_%d_%dx%dx%d_%dx%d" %
            (self.__class__.__name__, self.input.shape[0],
             self.sx, self.sy, self.n_channels,
             self.kx, self.ky), dtype=self.input.dtype)
        self.assign_kernel(self._kernel_name)

    def ocl_init(self):
        sh = self.output_shape
        self._gpu_init()
        self._global_size = [sh[3] * sh[2], sh[1] * sh[0]]
        self._local_size = None

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (int(numpy.ceil(self.output_size / block_size)),
                             1, 1)
        self._local_size = (block_size, 1, 1)

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

    def _gpu_run(self):
        self.unmap_vectors(self.input, self.output)
        self.execute_kernel(self._global_size, self._local_size)

    def ocl_run(self):
        self._gpu_run()

    def cuda_run(self):
        self._gpu_run()

    def cpu_run(self):
        self.input.map_read()
        self.output.map_invalidate()
        for batch, ch, out_x, out_y in product(*map(range, (
                self.input_batch_size, self.n_channels) + self.out_sxy)):
            x1 = out_x * self.sliding[0]
            y1 = out_y * self.sliding[1]
            test_idx = x1 + self.kx
            x2 = test_idx if test_idx <= self.sx else self.sx
            test_idx = y1 + self.ky
            y2 = test_idx if test_idx <= self.sy else self.sy
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

    MAPPING = set()

    def __init__(self, workflow, **kwargs):
        super(OffsetPooling, self).__init__(workflow, **kwargs)
        self.input_offset = formats.Vector()
        self.demand("input")

    def initialize(self, device, **kwargs):
        super(OffsetPooling, self).initialize(device=device, **kwargs)

        if self._no_output:
            return
        if not self.input_offset:
            self.input_offset.reset(numpy.zeros(self.output.shape,
                                                dtype=numpy.int32))
        else:
            assert self.input_offset.shape == self.output.shape
        self.input_offset.initialize(self.device)

    def set_args(self, *args):
        super(OffsetPooling, self).set_args(self.input, self.output,
                                            self.input_offset, *args)

    def ocl_run(self):
        self.input_offset.unmap()
        super(OffsetPooling, self).ocl_run()

    def cuda_run(self):
        self.input_offset.unmap()
        super(OffsetPooling, self).cuda_run()

    def cpu_run(self):
        self.input_offset.map_invalidate()
        super(OffsetPooling, self).cpu_run()

    def cpu_run_cut(self, cut, coords):
        batch, y1, x1, ch, out_y, out_x = coords
        cut_index = self.cpu_run_cut_offset(
            cut, numpy.ravel_multi_index((batch, out_y, out_x, ch),
                                         self.output.shape))
        i, j = numpy.unravel_index(cut_index, cut.shape)
        idx = numpy.ravel_multi_index((batch, y1 + i, x1 + j, ch),
                                      self.input.shape)
        val = numpy.ravel(self.input.mem)[idx]
        self.input_offset.mem[batch, out_y, out_x, ch] = idx
        return val


class MaxPoolingBase(OffsetPooling):
    """MaxPooling forward propagation base class.
    """
    MAPPING = set()

    def init_unpickled(self):
        super(MaxPoolingBase, self).init_unpickled()
        self._kernel_name = "max_pooling"

    def ocl_init(self):
        super(MaxPoolingBase, self).ocl_init()
        self.set_args()

    def cuda_init(self):
        super(MaxPoolingBase, self).cuda_init()
        self.set_args()


class MaxPooling(MaxPoolingBase):
    """MaxPooling forward propagation.
    """

    MAPPING = {"max_pooling"}

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

    MAPPING = {"maxabs_pooling"}

    def __init__(self, workflow, **kwargs):
        super(MaxAbsPooling, self).__init__(workflow, **kwargs)
        self.sources_["pooling"] = {"ABS_VALUES": 1}

    def cpu_run_cut_offset(self, cut, index):
        return numpy.abs(cut).argmax()


class StochasticPoolingBase(OffsetPooling):
    """Stochastic pooling forward propagation.

    Attributes:
        uniform: instance of veles.prng.Uniform.
    """
    MAPPING = set()

    def __init__(self, workflow, **kwargs):
        super(StochasticPoolingBase, self).__init__(workflow, **kwargs)
        self.uniform = kwargs.get("uniform")

    def init_unpickled(self):
        super(StochasticPoolingBase, self).init_unpickled()
        self._rand_set = False
        self._rand_arg = 3
        self._kernel_name = "stochastic_pooling"

    def initialize(self, device, **kwargs):
        super(StochasticPoolingBase, self).initialize(device=device, **kwargs)

        if self.uniform is None:
            self.uniform = Uniform(self)

        if self.uniform.output_bytes < (self.output_size << 1):
            if self.uniform.is_initialized:
                raise error.AlreadyExistsError(
                    "uniform is already initialized and "
                    "has not enough output size")
            self.uniform.output_bytes = self.output_size << 1

        self.uniform.initialize(self.device)

    def ocl_init(self):
        super(StochasticPoolingBase, self).ocl_init()
        self.assign_kernel(self._kernel_name)
        self.set_args()

    def cuda_init(self):
        super(StochasticPoolingBase, self).cuda_init()
        self.assign_kernel(self._kernel_name)
        self.set_args()

    def add_ref(self, unit):
        pass

    def cpu_run(self):
        self.uniform.cpu_fill(self.output_size << 1)
        super(StochasticPoolingBase, self).cpu_run()

    def ocl_run(self):
        if not self._rand_set:
            self.set_arg(self._rand_arg, self.uniform.output)
            self._rand_set = True
        self.uniform.ocl_fill(self.output_size << 1)
        super(StochasticPoolingBase, self).ocl_run()

    def cuda_run(self):
        if not self._rand_set:
            self.set_arg(self._rand_arg, self.uniform.output)
            self._rand_set = True
        self.uniform.cuda_fill(self.output_size << 1)
        super(StochasticPoolingBase, self).cuda_run()

    def calculate_position_cpu(self, index, vsum):
        rnd = self.uniform.output.mem.view(dtype=numpy.uint16)[index]
        return rnd * vsum / 65536

    def calculate_random_index_cpu(self, cut, index):
        rnd = self.uniform.output.mem.view(dtype=numpy.uint16)[index]
        return int(rnd * cut.size >> 16)


class StochasticPooling(StochasticPoolingBase):
    """StochasticPooling forward propagation.
    """

    MAPPING = {"stochastic_pooling"}

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

    MAPPING = {"stochastic_abs_pooling"}

    def __init__(self, workflow, **kwargs):
        super(StochasticAbsPooling, self).__init__(workflow, **kwargs)
        self.sources_["pooling"] = {"ABS_VALUES": 1}

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

    MAPPING = {"stochastic_pool_depool"}

    def __init__(self, workflow, **kwargs):
        super(StochasticPoolingDepooling, self).__init__(workflow, **kwargs)
        self._no_output = True

    def init_unpickled(self):
        super(StochasticPoolingDepooling, self).init_unpickled()
        self.sources_["pooling"]["USE_POOLING_DEPOOLING"] = 1
        self._rand_arg = 1
        self._kernel_name = "stochastic_pooling_depooling"

    def set_args(self, *args):
        self.set_arg(0, self.input)

    def cpu_run(self):
        raise RuntimeError("Not implemented")


class StochasticAbsPoolingDepooling(StochasticPoolingDepooling):
    """Stochastic abs pooling with depooling in-place.
    """

    MAPPING = {"stochastic_abs_pool_depool"}

    def __init__(self, workflow, **kwargs):
        super(StochasticAbsPoolingDepooling, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(StochasticAbsPoolingDepooling, self).init_unpickled()
        self.sources_["pooling"]["ABS_VALUES"] = 1


class AvgPooling(Pooling):
    """AvgPooling forward propagation.

    Must be assigned before initialize():

    Updates after run():

    Creates within initialize():

    """

    MAPPING = {"avg_pooling"}

    def init_unpickled(self):
        super(AvgPooling, self).init_unpickled()
        self._kernel_name = "avg_pooling"

    def initialize(self, device, **kwargs):
        super(AvgPooling, self).initialize(device=device, **kwargs)
        if self.device is None:
            return

    def ocl_init(self):
        super(AvgPooling, self).ocl_init()
        self.set_args(self.input, self.output)

    def cuda_init(self):
        super(AvgPooling, self).cuda_init()
        self.set_args(self.input, self.output)

    def cpu_run_cut(self, cut, coords):
        return numpy.sum(cut) / cut.size
