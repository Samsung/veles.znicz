# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jul 1, 2014

Deconvolutional layer.

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

import cuda4py.blas as cublas
import numpy
from zope.interface import implementer

from veles.config import root
from veles.compat import from_none
from veles.accelerated_units import IOpenCLUnit, ICUDAUnit, INumpyUnit
from veles.memory import Array
import veles.ocl_blas as ocl_blas
from veles.znicz.conv import ConvolutionalBase
import veles.znicz.nn_units as nn_units
from veles.distributable import TriviallyDistributable


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class Deconv(TriviallyDistributable, ConvolutionalBase, nn_units.Forward):
    # TriviallyDistributable overrides nn_units.Forward IDistributable
    """Deconvolutional layer for simple convolutional layer
    with linear activation and without bias.

    Must be assigned before initialize():
        input
        weights
        output_shape_source

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: input as batch of multichannel interleaved images.
        output: output as batch of multichannel interleaved images.
        weights: matrix of weights.
        output_shape_source: Array to get output shape from.
        n_kernels: number of convolutional kernels
                   in the corresponding convolutional layer.
        kx: kernel width.
        ky: kernel height.
        sliding: tuple of kernel sliding (by x-axis, by y-axis),
                 kx, ky MUST be a multiple of sliding to avoid irregularities.
        padding: tuple of virtual sample padding (left, top, right, bottom),
                 will be computed automatically based on sliding.
        weights_transposed: assume weights matrix as a transposed one.
        unsafe_padding: flag to enable unsafe padding and/or sliding.
    """

    MAPPING = {"deconv"}

    @staticmethod
    def compute_padding(sx, sy, kx, ky, sliding):
        """Computes required padding.
        """
        return (kx - sliding[1], ky - sliding[0],
                kx - sx % sliding[1] if sx % sliding[1] != 0
                else kx - sliding[1],
                ky - sy % sliding[0] if sy % sliding[0] != 0
                else ky - sliding[0])

    @staticmethod
    def check_padding_is_safe(kx, ky, sliding):
        if sliding[0] > (ky >> 1) or sliding[1] > (kx >> 1):
            raise ValueError(
                "sliding should not be greater than half of the kernel size")
        if kx % sliding[0] != 0 or kx % sliding[1] != 0:
            raise ValueError(
                "Kernel size should be multiple of sliding")

    def __init__(self, workflow, **kwargs):
        super(Deconv, self).__init__(workflow, **kwargs)
        self.unsafe_padding = kwargs.get("unsafe_padding", False)
        self.hits = Array()
        self.krn_clear_output_ = None
        self._global_size = None
        self._local_size = None
        del self.bias
        self.demand("n_kernels", "kx", "ky", "padding", "sliding",
                    "input", "weights", "output_shape_source")

    def init_unpickled(self):
        super(Deconv, self).init_unpickled()
        self.sources_["deconv/forward"] = {}

    def initialize(self, device, **kwargs):
        super(Deconv, self).initialize(device, **kwargs)

        self._dtype = self.input.dtype

        self.weights_shape = (tuple(reversed(self.weights.shape))
                              if self.weights_transposed
                              else self.weights.shape)

        if hasattr(self, "bias"):
            raise ValueError("bias should not be set")
        if (len(self.input.shape) != 4 or
                self.input.shape[3] != self.n_kernels):
            raise ValueError("Incorrectly shaped input encountered")
        if (len(self.weights_shape) != 2 or
                self.weights_shape[0] != self.n_kernels or
                self.weights_shape[1] % (self.kx * self.ky) != 0):
            raise ValueError("Incorrectly shaped weights encountered")

        output_shape = tuple(self.output_shape_source.shape)
        if len(output_shape) != 4:
            raise ValueError("Incorrect output_shape_source shape")
        if output_shape[0] != self.input.shape[0]:
            raise ValueError(
                "output_shape_source.shape[0] != input.shape[0]")

        try:
            self.check_padding_is_safe(self.kx, self.ky, self.sliding)
        except ValueError as e:
            if not self.unsafe_padding:
                raise from_none(e)
            self.warning("The padding will be unsafe")
            self._create_hits(output_shape)

        padding = Deconv.compute_padding(
            output_shape[2], output_shape[1], self.kx, self.ky, self.sliding)
        if self.padding is None:  # pylint: disable=E0203
            self.padding = padding
        elif self.padding != padding:
            if not self.unsafe_padding:
                raise ValueError(
                    "Expected padding %s but got %s" % (padding, self.padding))
            self._create_hits(output_shape)

        if not self.output:
            self.output.reset(numpy.zeros(output_shape,
                                          dtype=self._dtype))
        else:
            assert self.output.shape == output_shape

        self._output_shape = output_shape

        self._sy, self._sx, self._n_channels = self._output_shape[1:]
        self._kernel_size = self.kx * self.ky * self._n_channels

        self._kernel_app_per_image = self.input.sample_size // self.n_kernels
        self._kernel_app_total = (self._kernel_app_per_image *
                                  self.input.shape[0])

        self.init_vectors(self.input, self.weights, self.output, self.hits)

    def _create_hits(self, output_shape):
        if not self.hits:
            self.hits.reset(
                numpy.zeros(output_shape, dtype=numpy.int32))
        else:
            assert self.hits.size == int(numpy.prod(output_shape))

    def _gpu_init(self, blas_class):
        defines = {
            "USE_ATOMICS": 1,
            "WEIGHTS_TRANSPOSED": int(self.weights_transposed),
            "BATCH": self._output_shape[0],
            "SX": self._sx,
            "SY": self._sy,
            "N_CHANNELS": self._n_channels,
            "KX": self.kx,
            "KY": self.ky,
            "N_KERNELS": self.n_kernels,
            "PAD_LEFT": self.padding[0],
            "PAD_TOP": self.padding[1],
            "PAD_RIGHT": self.padding[2],
            "PAD_BOTTOM": self.padding[3],
            "SLIDE_X": self.sliding[0],
            "SLIDE_Y": self.sliding[1],
            "USE_HITS": int(bool(self.hits)),
            "DECONV_MODE": int(bool(self.hits)) + 1,
            "OUTPUT_SIZE": self.output.size
        }

        self.build_program(
            defines, "%s/%s_%d_%dx%dx%d_%dx%d_%d" % (
                root.common.dirs.cache, self.__class__.__name__,
                self.input.shape[0],
                self._output_shape[2], self._output_shape[1],
                self._output_shape[3],
                self.kx, self.ky, self.n_kernels), dtype=self._dtype)

        self.krn_pack_ = self.get_kernel("DirectPack")
        unpack_bytes = (self._kernel_app_per_image * self.unpack_size *
                        self._kernel_size * self.input.itemsize)
        self.device.request_temp_buffer(unpack_bytes)

        if self.hits:
            self.krn_pack_.set_arg(3, self.hits.devmem)

            self.krn_apply_hits_ = self.get_kernel("apply_hits")
            self.krn_apply_hits_.set_args(self.output.devmem, self.hits.devmem)

        self.gemm_ = blas_class.gemm(self._dtype)
        self.np_one = numpy.ones(1, dtype=self._dtype)
        self.np_zero = numpy.zeros(1, dtype=self._dtype)
        self._const_i = numpy.zeros(1, dtype=numpy.int64)

    def ocl_init(self):
        ocl_blas.OCLBLAS.attach_to_device(self.device)
        self._gpu_init(ocl_blas.OCLBLAS)

        self._global_size_pack = lambda size: (size,)
        self._local_size_pack = None

        if self.hits:
            self.krn_clear_hits_ = self.get_kernel("clear_hits")
            self.krn_clear_hits_.set_arg(0, self.hits.devmem)

            self._global_size_hits = (self.output.size,)
            self._local_size_hits = None

        self.krn_clear_output_ = self.get_kernel("clear_output")
        self.krn_clear_output_.set_arg(0, self.output.devmem)

        self._clear_output = lambda: (
            self.execute_kernel((self.output.size,), None,
                                self.krn_clear_output_))
        self._clear_hits = lambda: (
            self.execute_kernel((self.hits.size,), None, self.krn_clear_hits_))

        self._process_subblock = self._ocl_process_subblock

        self.krn_pack_.set_arg(1, self.output.devmem)

    def cuda_init(self):
        self._gpu_init(cublas.CUBLAS)

        block_size = self.device.suggest_block_size(self.krn_pack_)
        self._global_size_pack = (
            lambda size: (int(numpy.ceil(size / block_size)), 1, 1))
        self._local_size_pack = (block_size, 1, 1)

        if self.hits:
            block_size = self.device.suggest_block_size(self.krn_apply_hits_)
            self._global_size_hits = (
                int(numpy.ceil(self.output.size / block_size)), 1, 1)
            self._local_size_hits = (block_size, 1, 1)

        self._clear_output = lambda: self.output.devmem.memset32_async()
        self._clear_hits = lambda: self.hits.devmem.memset32_async()

        self._process_subblock = self._cuda_process_subblock

    def ocl_run(self):
        self.gpu_run()

    def cuda_run(self):
        self.gpu_run()

    def gpu_run(self):
        self.unmap_vectors(self.output, self.input, self.weights)
        unpack_data = self.device.get_temp_buffer()
        self._clear_output()
        if self.hits:
            self.hits.unmap()
            self._clear_hits()
        batch_size = self.output.shape[0]
        for i in range(0, batch_size, self.unpack_size):
            self._process_subblock(i, min(batch_size - i, self.unpack_size),
                                   unpack_data)
        if self.hits:
            self.execute_kernel(self._global_size_hits, self._local_size_hits,
                                self.krn_apply_hits_)

    def _cuda_process_subblock(self, start_image, image_count, unpack_data):
        output_offs = (start_image * self.input.sample_size *
                       self.input.itemsize)
        unpack_side = self._kernel_app_per_image * image_count

        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_T if self.weights_transposed
            else cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
            self._kernel_size, unpack_side, self.weights_shape[0],
            self.np_one, self.weights.devmem,
            int(self.input.devmem) + output_offs,
            self.np_zero, unpack_data)

        self.krn_pack_.set_arg(0, unpack_data)
        self.krn_pack_.set_arg(
            1, int(self.output.devmem) +
            start_image * self.output.sample_size * self.output.itemsize)
        limit = unpack_side * self._kernel_size
        self._const_i[0] = limit
        self.krn_pack_.set_arg(2, self._const_i)
        self.execute_kernel(self._global_size_pack(limit),
                            self._local_size_pack, self.krn_pack_)

    def _ocl_process_subblock(self, start_image, image_count, unpack_data):
        output_offs = start_image * self.input.sample_size
        unpack_side = self._kernel_app_per_image * image_count

        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_T if self.weights_transposed
            else cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
            self._kernel_size, unpack_side, self.weights_shape[0],
            self.np_one, self.weights.devmem,
            self.input.devmem,
            self.np_zero, unpack_data, offsetB=output_offs)

        self.krn_pack_.set_arg(0, unpack_data)
        self._const_i[0] = start_image * self.output.sample_size
        self.krn_pack_.set_arg(2, self._const_i)
        limit = unpack_side * self._kernel_size
        self.execute_kernel(self._global_size_pack(limit),
                            self._local_size_pack, self.krn_pack_)

    def numpy_run(self):
        raise NotImplementedError()
