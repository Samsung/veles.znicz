# encoding: utf-8
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on April 23, 2013

A layer for local response normalization.
Detailed description given in article by Krizhevsky, Sutskever and Hinton:
"ImageNet Classification with Deep Convolutional Neural Networks"

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


from __future__ import division

import numpy
from zope.interface import implementer

from veles.znicz.nn_units import AcceleratedUnit, Forward, GradientDescentBase
from veles.accelerated_units import IOpenCLUnit, ICUDAUnit


class LocalResponseNormalizer(AcceleratedUnit):
    hide_from_registry = True
    """
    A base class for forward and backward units of local
    response normalization.
    """
    def __init__(self, workflow, **kwargs):
        self.alpha = kwargs.get("alpha", 0.0001)
        self.beta = kwargs.get("beta", 0.75)
        self.k = kwargs.get("k", 2)
        self.n = kwargs.get("n", 5)
        self._num_of_chans = None

        super(LocalResponseNormalizer, self).__init__(workflow, **kwargs)

    def _subsums(self, source_array, window_size):
        """
        For each channel calculates the sum of its neighbour channels.
        source_array must be a 4-dimensional array (channel dim is the last).
        """
        assert len(source_array.shape) == 4
        subsums = numpy.empty_like(source_array)
        num_of_chans = source_array.shape[3]
        for i in range(num_of_chans):
            min_index = max(0, i - int(window_size / 2))
            max_index = min(i + int(window_size / 2), num_of_chans - 1)
            array_slice = source_array[:, :, :, min_index: max_index + 1]
            subsums[:, :, :, i] = numpy.sum(array_slice, axis=3)
        return subsums

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


@implementer(IOpenCLUnit, ICUDAUnit)
class LRNormalizerForward(LocalResponseNormalizer, Forward):
    """
    Forward propagation of local response normalization.
    """

    MAPPING = {"norm"}

    def init_unpickled(self):
        super(LRNormalizerForward, self).init_unpickled()
        self.sources_["normalization"] = {}

    def initialize(self, device, **kwargs):
        super(LRNormalizerForward, self).initialize(device, **kwargs)

        if not self.output:
            self.output.reset(numpy.zeros_like(self.input.mem))
        else:
            assert self.output.shape == self.input.shape

        self._num_of_chans = self.input.mem.shape[3]
        self.init_vectors(self.input, self.output)

    def _gpu_init(self):
        defines = {"ALPHA": self.alpha, "BETA": self.beta, "K": self.k,
                   "N": self.n, "NUM_OF_CHANS": self._num_of_chans,
                   "OUTPUT_SIZE": self.output.size // self._num_of_chans}

        self.build_program(defines, "%s_%s" %
                           (self.__class__.__name__,
                            "x".join(str(x) for x in self.input.shape)),
                           dtype=self.input.dtype)
        self.assign_kernel("forward")
        self.set_args(self.input, self.output)

    def ocl_init(self):
        self._gpu_init()
        self._global_size = [self.output.size // self._num_of_chans]
        self._local_size = None

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (
            int(numpy.ceil((self.output.size // self._num_of_chans) /
                           block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def cpu_run(self):
        self.output.map_invalidate()
        self.input.map_read()

        assert len(self.input.shape) == 4
        input_squared = numpy.square(self.input.mem)
        subsums = self._subsums(input_squared, self.n)
        subsums *= self.alpha
        subsums += self.k
        subsums **= self.beta

        numpy.copyto(self.output.mem, self.input.mem / subsums)

    def _gpu_run(self):
        self.unmap_vectors(self.input, self.output)
        self.execute_kernel(self._global_size, self._local_size)

    def ocl_run(self):
        self._gpu_run()

    def cuda_run(self):
        self._gpu_run()

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


@implementer(IOpenCLUnit, ICUDAUnit)
class LRNormalizerBackward(LocalResponseNormalizer, GradientDescentBase):
    """
    Backward-propagation for local response normalization.
    """

    MAPPING = {"norm"}

    def init_unpickled(self):
        super(LRNormalizerBackward, self).init_unpickled()
        self.sources_["normalization"] = {}

    def initialize(self, device, **kwargs):
        self._num_of_chans = self.input.mem.shape[3]
        super(LRNormalizerBackward, self).initialize(device, **kwargs)

    def _gpu_init(self):
        defines = {"ALPHA": self.alpha, "BETA": self.beta, "K": self.k,
                   "N": self.n, "NUM_OF_CHANS": self._num_of_chans,
                   "OUTPUT_SIZE": self.err_input.size // self._num_of_chans}

        self.build_program(defines, "%s_%s" %
                           (self.__class__.__name__,
                            "x".join(str(x) for x in self.err_output.shape)),
                           dtype=self.input.dtype)
        self.assign_kernel("backward")
        self.set_args(self.err_output, self.input, self.err_input)

    def ocl_init(self):
        self._gpu_init()
        self._global_size = [self.err_input.size // self._num_of_chans]
        self._local_size = None

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (
            int(numpy.ceil((self.err_input.size // self._num_of_chans) /
                           block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def cpu_run(self):
        self.err_input.map_invalidate()
        self.err_output.map_read()
        self.input.map_read()

        assert len(self.input.shape) == 4
        assert self.input.shape == self.err_output.shape

        num_of_chans = self.input.shape[3]

        input_squared = numpy.square(self.input.mem)
        input_subsums = self._subsums(input_squared, self.n)

        input_subsums *= self.alpha
        input_subsums += self.k

        input_subsums_powered = numpy.power(input_subsums, (self.beta + 1))

        err_h = self.err_input.mem
        err_y = self.err_output.mem

        for i in range(num_of_chans):
            min_index = max(0, i - int(self.n / 2))
            max_index = min(i + int(self.n / 2), num_of_chans - 1)

            delta_h = numpy.zeros(dtype=numpy.float64,
                                  shape=err_h[:, :, :, i].shape)
            for j in range(min_index, max_index + 1):
                dh = numpy.zeros(shape=delta_h.shape, dtype=numpy.float64)
                if i == j:
                    dh += input_subsums[:, :, :, j]
                dh -= (2 * self.beta * self.alpha *
                       self.input.mem[:, :, :, i] *
                       self.input.mem[:, :, :, j])
                dh *= (err_y[:, :, :, j] /
                       input_subsums_powered[:, :, :, j])
                delta_h += dh
            numpy.copyto(err_h[:, :, :, i], delta_h)

    def _gpu_run(self):
        self.unmap_vectors(self.err_output, self.input, self.err_input)
        self.execute_kernel(self._global_size, self._local_size)

    def ocl_run(self):
        self._gpu_run()

    def cuda_run(self):
        self._gpu_run()

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
