# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jul 2, 2014

Gradient descent for deconvolutional layer.

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
from veles.compat import from_none

from veles.accelerated_units import IOpenCLUnit, ICUDAUnit, INumpyUnit
import veles.ocl_blas as ocl_blas
import veles.znicz.nn_units as nn_units
from veles.znicz.conv import ConvolutionalBase
from veles.znicz.deconv import Deconv


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class GDDeconv(ConvolutionalBase, nn_units.GradientDescentBase):
    """Gradient Descent.

    Must be assigned before initialize():
        input
        err_output
        weights

    Updates after run():
        err_input
        weights

    Creates within initialize():
        err_input

    Attributes:
        krn_err_input_clear_: OpenCL kernel for setting err_input with zeros.
        krn_err_input_: OpenCL kernel for computing err_input.
        krn_weights_: OpenCL kernel for weights update.
        n_kernels: number of convolutional kernels.
        kx: kernel width.
        ky: kernel height.
        padding: padding.
        sliding: sliding.
    """

    MAPPING = {"deconv"}

    def __init__(self, workflow, **kwargs):
        super(GDDeconv, self).__init__(workflow, **kwargs)
        self.cl_const = None
        self._global_size_err_input = None
        self._local_size_err_input = None
        self._global_size_weights = None
        self._local_size_weights = None
        self.reduce_size = self.REDUCE_SIZE
        self.hits = None
        self.krn_err_output_ = None
        self.krn_err_input_ = None
        self.krn_weights_ = None
        self.krn_compute_col_sums_ = None
        self.demand("weights")

    @property
    def channels_number(self):
        sy, sx = self.err_output.shape[1:3]
        return self.err_output.size // (self.err_output.shape[0] * sx * sy)

    @property
    def weights_number(self):
        return self.n_kernels * self.kx * self.ky * self.channels_number

    @property
    def ky_kx(self):
        return self.err_output.mem.shape[1:3]

    @property
    def unsafe_padding(self):
        """hits implies unsafe_padding.
        """
        return bool(self.hits)

    def initialize(self, device, **kwargs):
        super(GDDeconv, self).initialize(device, **kwargs)

        if self.bias is not None:
            raise ValueError("bias should not be set")
        if (len(self.weights_shape) != 2 or
                self.weights_shape[0] != self.n_kernels or
                self.weights_shape[1] % (self.kx * self.ky) != 0):
            raise ValueError(
                "Incorrectly shaped weights encountered")
        if (len(self.input.shape) != 4 or
                self.input.shape[3] != self.n_kernels):
            raise ValueError(
                "Incorrectly shaped input encountered")
        if (len(self.err_output.shape) != 4 or
                self.err_output.shape[0] != self.input.shape[0]):
            raise ValueError(
                "Incorrectly shaped err_output encountered")

        sy, sx = self.ky_kx

        if self.weights.size != self.weights_number:
            raise ValueError(
                "Expected number of weights to match "
                "input, n_kernels, kx, ky parameters")

        try:
            Deconv.check_padding_is_safe(self.kx, self.ky, self.sliding)
        except ValueError as e:
            if not self.hits:
                raise from_none(e)
            self.warning("The padding will be unsafe")
        padding = Deconv.compute_padding(
            sx, sy, self.kx, self.ky, self.sliding)
        if self.padding is None:  # pylint: disable=E0203
            self.padding = padding
        elif self.padding != padding and not self.unsafe_padding:
            raise ValueError(
                "Expected padding %s got %s"
                % (str(padding), str(self.padding)))
        if self.hits:
            self.hits.initialize(self.device)

        self._batch_size = self.err_output.shape[0]
        self._kernel_app_per_image = self.input.sample_size // self.n_kernels
        self._kernel_app_total = (self._kernel_app_per_image *
                                  self.input.shape[0])
        self._kernel_size = self.kx * self.ky * self.channels_number

    def _gpu_init(self, blas_class):
        dtype = self.err_output.dtype
        self._weights_const = numpy.zeros(16, dtype=dtype)

        self.sources_["conv/forward"] = {}
        self.sources_["deconv/gradient_descent/weights_update"] = {
            "USE_ORTHO": int(bool(self.factor_ortho)),
            'USE_MOMENT': int(bool(self.gradient_moment))
        }

        sy, sx = self.ky_kx

        side = self.weights_shape[0]
        other = self.weights.size // side
        self._other = other
        assert side == self.n_kernels
        assert other == self.kx * self.ky * self.channels_number

        defines = {
            'INCLUDE_BIAS': 0,
            'H': other,
            'Y': side,
            'ACTIVATION_LINEAR': 1,
            'APPLY_GRADIENT': int(self.apply_gradient),
            'WEIGHTS_TRANSPOSED': int(self.weights_transposed),
            'ACCUMULATE_GRADIENT': int(self.accumulate_gradient),
            'BATCH': self._batch_size,
            'SX': sx,
            'SY': sy,
            'N_CHANNELS': self.channels_number,
            'KX': self.kx,
            'KY': self.ky,
            'N_KERNELS': self.n_kernels,
            'PAD_LEFT': self.padding[0],
            'PAD_TOP': self.padding[1],
            'PAD_RIGHT': self.padding[2],
            'PAD_BOTTOM': self.padding[3],
            'SLIDE_X': self.sliding[0],
            'SLIDE_Y': self.sliding[1],
            'REDUCE_SIZE': self.reduce_size,
            'USE_HITS': int(bool(self.hits)),
            'DECONV_MODE': int(bool(self.hits)) + 1,
            'OUTPUT_SIZE': self.err_output.size
        }

        self.build_program(
            defines, "%s_%d_%d_%d" % (
                self.__class__.__name__,
                self.input.shape[0],
                self.input.sample_size,
                self.err_output.sample_size),
            dtype=dtype)

        self.krn_err_output_ = self.get_kernel("err_output_update")
        self.krn_err_output_.set_arg(0, self.err_output.devmem)
        if self.hits:
            self.krn_err_output_.set_arg(1, self.hits.devmem)

        if self.need_gradient_weights:
            self.krn_weights_ = self.get_kernel("weights_update")
            self.krn_weights_.set_args(
                self.weights.devmem, self.gradient_weights.devmem,
                self.accumulated_gradient_weights.devmem,
                self.gradient_weights_with_moment.devmem)

        if self.need_gradient_weights and self.factor_ortho:
            self.krn_compute_col_sums_ = self.get_kernel("compute_col_sums")
            self.krn_compute_col_sums_.set_args(self.weights.devmem,
                                                self.col_sums.devmem)
            self.krn_weights_.set_arg(13, self.col_sums.devmem)

        self.gemm_ = blas_class.gemm(dtype)
        self.np_one = numpy.ones(1, dtype=dtype)
        self.np_zero = numpy.zeros(1, dtype=dtype)
        self._const_i = numpy.zeros(1, dtype=numpy.int64)
        self.np_err_input_alpha = numpy.ones(1, dtype=dtype)
        self.np_err_input_beta = numpy.zeros(1, dtype=dtype)

        self.assign_kernel("Unpack1D")
        unpack_bytes = (self._kernel_app_per_image * self.unpack_size *
                        self._kernel_size * self.err_output.itemsize)
        self.device.request_temp_buffer(unpack_bytes)

    def ocl_init(self):
        ocl_blas.OCLBLAS.attach_to_device(self.device)
        self._gpu_init(ocl_blas.OCLBLAS)

        if self.need_gradient_weights:
            self._global_size_ortho = (self._other * self.reduce_size,)
            self._local_size_ortho = (self.reduce_size,)

        self._global_size_err_output = (self.err_output.size,)
        self._local_size_err_output = None

        self._global_size_unpack = lambda size: (size,)
        self._local_size_unpack = None

        if self.need_gradient_weights:
            self._global_size_weights = (self.weights.size,)
            self._local_size_weights = None

        self._process_subblock = self._ocl_process_subblock

        self._kernel_.set_arg(0, self.err_output.devmem)

    def cuda_init(self):
        self._gpu_init(cublas.CUBLAS)

        if self.need_gradient_weights:
            self._global_size_ortho = (self._other, 1, 1)
            self._local_size_ortho = (self.reduce_size, 1, 1)

        block_size = self.device.suggest_block_size(self.krn_err_output_)
        self._global_size_err_output = (
            int(numpy.ceil(self.err_output.size / block_size)), 1, 1)
        self._local_size_err_output = (block_size, 1, 1)

        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size_unpack = (
            lambda size: (int(numpy.ceil(size / block_size)), 1, 1))
        self._local_size_unpack = (block_size, 1, 1)

        if self.need_gradient_weights:
            block_size = self.device.suggest_block_size(self.krn_weights_)
            self._global_size_weights = (
                int(numpy.ceil(self.weights.size / block_size)), 1, 1)
            self._local_size_weights = (block_size, 1, 1)

        self._process_subblock = self._cuda_process_subblock

    def gpu_err_output_update(self):
        self.err_output.unmap()
        self.execute_kernel(
            self._global_size_err_output, self._local_size_err_output,
            self.krn_err_output_)

    def ocl_run(self):
        self.gpu_run()

    def cuda_run(self):
        self.gpu_run()

    def gpu_run(self):
        # Divide err_output by hits count
        self.gpu_err_output_update()

        # Update err_input and simultaneousely accumulate gradient
        self.unmap_vectors(self.err_input, self.weights, self.err_output,
                           self.gradient_weights)
        unpack_data = self.device.get_temp_buffer()
        for i in range(0, self._batch_size, self.unpack_size):
            self._process_subblock(
                i, min(self._batch_size - i, self.unpack_size), unpack_data)

        # Update weights
        self.gpu_weights_update()

    def _cuda_process_subblock(self, start_image, image_count, unpack_data):
        self._kernel_.set_arg(
            0, int(self.err_output.devmem) +
            start_image * self.err_output.sample_size *
            self.err_output.itemsize)
        self._kernel_.set_arg(1, unpack_data)
        unpack_side = self._kernel_app_per_image * image_count
        limit = unpack_side * self._kernel_size
        self._const_i[0] = limit
        self._kernel_.set_arg(2, self._const_i)
        self.execute_kernel(self._global_size_unpack(limit),
                            self._local_size_unpack)
        output_offs = (start_image * self.input.sample_size *
                       self.input.itemsize)

        # Update err_input
        if self.need_err_input:
            self.np_err_input_alpha[0] = self.err_input_alpha
            self.np_err_input_beta[0] = self.err_input_beta
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N if self.weights_transposed
                else cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
                self.weights_shape[0], unpack_side, self._kernel_size,
                self.np_err_input_alpha, self.weights.devmem, unpack_data,
                self.np_err_input_beta,
                int(self.err_input.devmem) + output_offs)

        if not self.need_gradient_weights:
            return

        # Accumulate gradient
        if self.weights_transposed:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.n_kernels, self._kernel_size, unpack_side,
                self.np_one, int(self.input.devmem) + output_offs,
                unpack_data, self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem)
        else:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self._kernel_size, self.n_kernels, unpack_side, self.np_one,
                unpack_data, int(self.input.devmem) + output_offs,
                self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem)

    def _ocl_process_subblock(self, start_image, image_count, unpack_data):
        self._kernel_.set_arg(1, unpack_data)
        self._const_i[0] = start_image * self.err_output.sample_size
        self._kernel_.set_arg(2, self._const_i)
        unpack_side = self._kernel_app_per_image * image_count
        limit = unpack_side * self._kernel_size
        self.execute_kernel(self._global_size_unpack(limit),
                            self._local_size_unpack)
        output_offs = start_image * self.input.sample_size

        # Update err_input
        if self.need_err_input:
            self.np_err_input_alpha[0] = self.err_input_alpha
            self.np_err_input_beta[0] = self.err_input_beta
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N if self.weights_transposed
                else cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
                self.weights_shape[0], unpack_side, self._kernel_size,
                self.np_err_input_alpha, self.weights.devmem, unpack_data,
                self.np_err_input_beta, self.err_input.devmem,
                offsetC=output_offs)

        if not self.need_gradient_weights:
            return

        # Accumulate gradient
        if self.weights_transposed:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.n_kernels, self._kernel_size, unpack_side,
                self.np_one, self.input.devmem,
                unpack_data, self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem, offsetA=output_offs)
        else:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self._kernel_size, self.n_kernels, unpack_side,
                self.np_one, unpack_data, self.input.devmem,
                self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem, offsetB=output_offs)

    def numpy_run(self):
        raise NotImplementedError()
