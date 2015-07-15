# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 4, 2014

Cutter unit.

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

from veles.accelerated_units import (
    AcceleratedUnit, IOpenCLUnit, ICUDAUnit, INumpyUnit)

import veles.error as error
from veles.memory import Array, reshape
import veles.znicz.nn_units as nn_units
from veles.units import Unit


class CutterBase(Unit):
    hide_from_registry = True

    def __init__(self, workflow, padding, *args, **kwargs):
        super(CutterBase, self).__init__(workflow, *args, **kwargs)
        self.padding = padding
        self.demand("input")

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        if value is None:
            raise ValueError("padding may not be None")
        if type(value) not in (tuple, list):
            raise TypeError("padding must either of type tuple or list")
        if len(value) != 4:
            raise ValueError(
                "padding must be of length 4: (left, top, right, bottom)")
        self._padding = value

    def create_stuff(self, prefix):
        setattr(self, "_%s_origin" % prefix, (
            self.padding[0] * self.input.shape[3] * self.input.itemsize,
            self.padding[1], 0))
        self._region = (
            self.output_shape[2] * self.output_shape[3] * self.input.itemsize,
            self.output_shape[1], self.input.shape[0])
        setattr(self, "_%s_row_pitch" % prefix, (
            self.input.shape[2] * self.input.shape[3] * self.input.itemsize))
        setattr(self, "_%s_slice_pitch" % prefix,
                self.input.sample_size * self.input.itemsize)
        setattr(self, "_%s_slice_height" % prefix,
                self.input.shape[1])


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class Cutter(nn_units.Forward, CutterBase):
    MAPPING = {"cutter"}
    """Cuts rectangular area from an input.

    Must be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: input as batch of samples.
        output: output as batch of samples.
    """
    def __init__(self, workflow, **kwargs):
        super(Cutter, self).__init__(workflow, **kwargs)
        self.exports.append("padding")

    def initialize(self, device, **kwargs):
        if not self.input or len(self.input.shape) != 4:
            raise error.BadFormatError(
                "input should be assigned and have shape of 4: "
                "(n_samples, sy, sx, n_channels)")
        if self.padding[0] < 0 or self.padding[1] < 0:
            raise error.BadFormatError(
                "padding[0], padding[1] should not be less than zero")
        super(Cutter, self).initialize(device=device, **kwargs)

        shape = list(self.input.shape)
        shape[2] -= self.padding[0] + self.padding[2]
        shape[1] -= self.padding[1] + self.padding[3]
        if shape[2] <= 0 or shape[1] <= 0:
            raise error.BadFormatError("Resulted output shape is empty")
        self.output_shape = shape

        if not self.output:
            self.output.reset(numpy.zeros(self.output_shape, self.input.dtype))
        else:
            assert self.output.shape == self.output_shape

        for vec in self.input, self.output:
            vec.initialize(self.device)

        self.create_stuff("src")

    def ocl_init(self):
        pass

    def cuda_init(self):
        pass

    def ocl_run(self):
        """Forward propagation from batch on OpenCL.
        """
        self.unmap_vectors(self.output, self.input)
        self.device.queue_.copy_buffer_rect(
            self.input.devmem, self.output.devmem,
            self._src_origin, (0, 0, 0), self._region,
            self._src_row_pitch, self._src_slice_pitch,
            need_event=False)

    def cuda_run(self):
        """Forward propagation from batch on CUDA.
        """
        self.unmap_vectors(self.output, self.input)
        self.input.devmem.memcpy_3d_async(
            self._src_origin, (0, 0, 0), self._region,
            self._src_row_pitch, self._src_slice_height,
            dst=self.output.devmem)

    def numpy_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.output.map_invalidate()
        self.input.map_read()
        out = reshape(self.output.mem, self.output_shape)
        inp = self.input.mem
        out[:, :, :, :] = inp[
            :, self.padding[1]:self.padding[1] + self.output_shape[1],
            self.padding[0]:self.padding[0] + self.output_shape[2], :]


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class GDCutter(nn_units.GradientDescentBase, CutterBase):
    """Gradient descent for Cutter.
    """
    MAPPING = {"cutter"}

    def initialize(self, device, **kwargs):
        if not self.input or len(self.input.shape) != 4:
            raise error.BadFormatError(
                "input should be assigned and have shape of 4: "
                "(n_samples, sy, sx, n_channels)")
        if self.padding[0] < 0 or self.padding[1] < 0:
            raise error.BadFormatError(
                "padding[0], padding[1] should not be less than zero")
        if not self.err_output:
            raise error.BadFormatError("err_output should be assigned")
        super(GDCutter, self).initialize(device=device, **kwargs)

        sh = list(self.input.shape)
        sh[2] -= self.padding[0] + self.padding[2]
        sh[1] -= self.padding[1] + self.padding[3]
        if sh[2] <= 0 or sh[1] <= 0:
            raise error.BadFormatError("Resulted output shape is empty")
        output_size = int(numpy.prod(sh))
        if self.err_output.size != output_size:
            raise error.BadFormatError(
                "Computed err_output size differs from an assigned one")
        self.output_shape = sh

        if not self.err_input:
            self.err_input.reset(numpy.zeros_like(self.input.mem))
        else:
            assert self.err_input.shape == self.input.shape

        self.init_vectors(self.err_output, self.err_input)

        self.create_stuff("dst")

    def ocl_init(self):
        self.sources_["cutter"] = {}
        self.build_program(
            {}, "%s_%s_%s" %
            (self.__class__.__name__,
             "x".join(str(x) for x in self.err_input.shape),
             "x".join(str(x) for x in self.padding)),
            dtype=self.err_input.dtype)
        self.assign_kernel("clear_err_input")
        self.set_args(self.err_input)

    def cuda_init(self):
        pass

    def ocl_run(self):
        """Backward propagation from batch on OpenCL.
        """
        self.unmap_vectors(self.err_output, self.err_input)
        self.execute_kernel([self.err_input.size], None)
        self.device.queue_.copy_buffer_rect(
            self.err_output.devmem, self.err_input.devmem,
            (0, 0, 0), self._dst_origin, self._region,
            0, 0, self._dst_row_pitch, self._dst_slice_pitch,
            need_event=False)

    def cuda_run(self):
        """Backward propagation from batch on OpenCL.
        """
        self.unmap_vectors(self.err_output, self.err_input)
        self.err_input.devmem.memset32_async()
        self.err_output.devmem.memcpy_3d_async(
            (0, 0, 0), self._dst_origin, self._region,
            0, 0, self._dst_row_pitch, self._dst_slice_height,
            dst=self.err_input.devmem)

    def numpy_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.err_input.map_invalidate()
        self.err_output.map_read()
        out = reshape(self.err_output.mem, self.output_shape)
        inp = self.err_input.mem
        inp[:] = 0
        inp[:, self.padding[1]:self.padding[1] + self.output_shape[1],
            self.padding[0]:self.padding[0] + self.output_shape[2], :] = out


@implementer(ICUDAUnit, IOpenCLUnit)
class Cutter1D(AcceleratedUnit):
    """Cuts the specified interval from each 1D sample of input batch
    into output.

    y = alpha * x + beta * y
    """
    def __init__(self, workflow, **kwargs):
        super(Cutter1D, self).__init__(workflow, **kwargs)
        self.alpha = kwargs.get("alpha")
        self.beta = kwargs.get("beta")
        self.input_offset = kwargs.get("input_offset", -1)
        self.output_offset = kwargs.get("output_offset", 0)
        self.length = kwargs.get("length", -1)
        self.output = Array()
        self.demand("alpha", "beta", "input")

    def init_unpickled(self):
        super(Cutter1D, self).init_unpickled()
        self.sources_["cutter"] = {}

    def initialize(self, device, **kwargs):
        if self.input_offset < 0 or self.length < 0:
            return True
        super(Cutter1D, self).initialize(device, **kwargs)

        if not self.output:
            self.output.reset(
                numpy.zeros(
                    (self.input.shape[0], self.output_offset + self.length),
                    dtype=self.input.dtype))
        else:
            assert self.output.size >= self.output_offset + self.length

        self.init_vectors(self.input, self.output)

    def cuda_init(self):
        dtype = self.input.dtype
        itemsize = self.input.itemsize
        limit = self.input.shape[0] * self.length

        self.build_program({}, "%s" % self.__class__.__name__, dtype=dtype)
        self.assign_kernel("cutter_1d_forward")

        self.set_args(
            int(self.input.devmem) + self.input_offset * itemsize,
            numpy.array([self.alpha], dtype=dtype),
            numpy.array([self.input.sample_size], dtype=numpy.int32),
            int(self.output.devmem) + self.output_offset * itemsize,
            numpy.array([self.beta], dtype=dtype),
            numpy.array([self.output.sample_size], dtype=numpy.int32),
            numpy.array([self.length], dtype=numpy.int32),
            numpy.array([limit], dtype=numpy.int32))

        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (int(numpy.ceil(limit / block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def ocl_init(self):
        dtype = self.input.dtype

        self.build_program({}, "%s" % self.__class__.__name__, dtype=dtype)
        self.assign_kernel("cutter_1d_forward")

        self.set_args(
            self.input.devmem,
            numpy.array([self.input_offset], dtype=numpy.int32),
            numpy.array([self.alpha], dtype=dtype),
            numpy.array([self.input.sample_size], dtype=numpy.int32),
            self.output.devmem,
            numpy.array([self.output_offset], dtype=numpy.int32),
            numpy.array([self.beta], dtype=dtype),
            numpy.array([self.output.sample_size], dtype=numpy.int32))

        self._global_size = (self.input.shape[0], self.length)
        self._local_size = None

    def _gpu_run(self):
        self.unmap_vectors(self.input, self.output)
        self.execute_kernel(self._global_size, self._local_size)

    def cuda_run(self):
        return self._gpu_run()

    def ocl_run(self):
        return self._gpu_run()
