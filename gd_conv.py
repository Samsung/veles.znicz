# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 14, 2013

Gradient descent for convolutional units.

* :class:`GradientDescentConv` couples with :class:`veles.znicz.conv.Conv`
* :class:`GDTanhConv` couples with :class:`veles.znicz.conv.ConvTanh`
* :class:`GDRELUConv` couples with :class:`veles.znicz.conv.ConvRELU`
* :class:`GDStrictRELUConv` couples with \
    :class:`veles.znicz.conv.ConvStrictRELU`

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
from itertools import product
import numpy
from zope.interface import implementer

import veles.error as error
from veles.memory import reshape_transposed
from veles.accelerated_units import IOpenCLUnit, ICUDAUnit, INumpyUnit
import veles.ocl_blas as ocl_blas
from veles.znicz.conv import ConvolutionalBase
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class GradientDescentConv(ConvolutionalBase, nn_units.GradientDescentBase):
    """Gradient descent for simple convolutional layer (no activation).

    Must be assigned before initialize():
        output
        input
        err_output
        weights
        bias
        batch_size

    Updates after run():
        err_input
        err_output
        weights
        bias

    Creates within initialize():
        err_input

    Attributes:
        krn_err_input_clear_: OpenCL kernel for setting err_input with zeros.
        krn_err_input_: OpenCL kernel for computing err_input.
        krn_weights_: OpenCL kernel for weights update.
        krn_err_output_: OpenCL kernel for err_output update.
        krn_bias_: OpenCL kernel for bias update.
        n_kernels: number of convolutional kernels.
        kx: kernel width.
        ky: kernel height.
    """

    MAPPING = {"conv"}

    def __init__(self, workflow, **kwargs):
        super(GradientDescentConv, self).__init__(workflow, **kwargs)
        self.reduce_size = self.REDUCE_SIZE
        self.cl_const = None
        self.krn_err_input_clear_ = None
        self.krn_err_input_ = None
        self.krn_weights_ = None
        self.krn_err_output_ = None
        self.krn_bias_ = None
        self.krn_err_output_name = None
        self.demand("weights")
        if self.include_bias:
            self.demand("bias")

    def initialize(self, device, **kwargs):
        super(GradientDescentConv, self).initialize(device=device, **kwargs)

        self._batch_size = self.input.shape[0]
        self._sy = self.input.shape[1]
        self._sx = self.input.shape[2]
        self._n_channels = (self.input.size //
                            (self._batch_size * self._sx * self._sy))
        self._kernel_size = self.kx * self.ky * self._n_channels
        self._dtype = self.err_output.dtype
        self._kx_app = (
            1 + ((self._sx - self.kx +
                  self.padding[0] + self.padding[2]) // self.sliding[0]))
        self._ky_app = (
            1 + ((self._sy - self.ky +
                  self.padding[1] + self.padding[3]) // self.sliding[1]))
        self._kernel_app_per_image = self._kx_app * self._ky_app
        self._kernel_app_total = self._batch_size * self._kernel_app_per_image

        self.cl_const = numpy.zeros(9, dtype=self._dtype)

        self._side = self.weights_shape[0]
        self._other = self.weights.size // self._side
        assert self._side == self.n_kernels
        assert self._other == self.kx * self.ky * self._n_channels

        n_weights = self.n_kernels * self.kx * self.ky * self._n_channels
        if self.weights.size != n_weights:
            raise error.BadFormatError(
                "Expected number of weights to match "
                "input, n_kernels, kx, ky parameters")
        if self.include_bias and self.bias.size != self.n_kernels:
            raise error.BadFormatError("Expected bias to match n_kernels")
        if (self.input.size !=
                self._batch_size * self._sy * self._sx * self._n_channels):
            raise error.BadFormatError(
                "Expected input size to match "
                "batch_size * sy * sx * n_channels")

    def _gpu_init(self, blas_class):
        dtype = self.err_output.dtype
        self._weights_const = numpy.zeros(16, dtype=dtype)
        self._bias_const = numpy.zeros(16, dtype=dtype)

        self.sources_["conv/forward"] = {}
        self.sources_["conv/gradient_descent/err_input_update"] = {}
        self.sources_["all2all/gradient_descent/weights_update"] = {
            "USE_ORTHO": int(bool(self.factor_ortho)),
            "USE_MOMENT": int(bool(self.gradient_moment))
        }
        self.sources_["all2all/gradient_descent/bias_update"] = {
            "BIAS_SIZE": self.n_kernels,
            "OUTPUT_SIZE": self._kernel_app_total,
            "USE_MOMENT": int(bool(self.gradient_moment_bias))
        }

        defines = {
            'H': self._other,
            'Y': self._side,
            'APPLY_GRADIENT': int(self.apply_gradient),
            'WEIGHTS_TRANSPOSED': int(self.weights_transposed),
            'ACCUMULATE_GRADIENT': int(self.accumulate_gradient),
            'USE_ATOMICS': 1,
            'BATCH': self._batch_size,
            'SX': self._sx,
            'SY': self._sy,
            'N_CHANNELS': self._n_channels,
            'KX': self.kx,
            'KY': self.ky,
            'N_KERNELS': self.n_kernels,
            'PAD_LEFT': self.padding[0],
            'PAD_TOP': self.padding[1],
            'PAD_RIGHT': self.padding[2],
            'PAD_BOTTOM': self.padding[3],
            'SLIDE_X': self.sliding[0],
            'SLIDE_Y': self.sliding[1],
            'REDUCE_SIZE': self.reduce_size
        }

        self.build_program(defines, "%s_%d_%d_%d_%dx%dx%d" % (
            self.__class__.__name__, self.input.shape[0],
            self.input.sample_size, self.err_output.sample_size,
            self.kx, self.ky, self.n_kernels),
            dtype=self._dtype)

        if self.need_gradient_weights:
            self.krn_weights_ = self.get_kernel("weights_update")
            self.krn_weights_.set_args(
                self.weights.devmem, self.gradient_weights.devmem,
                self.accumulated_gradient_weights.devmem,
                self.gradient_weights_with_moment.devmem)

        if self.need_gradient_weights and self.include_bias:
            self.krn_bias_ = self.get_kernel("bias_update")
            self.krn_bias_.set_args(
                self.err_output.devmem, self.bias.devmem,
                self.gradient_bias.devmem,
                self.accumulated_gradient_bias.devmem,
                self.gradient_bias_with_moment.devmem)

        if self.need_gradient_weights and self.factor_ortho:
            self.krn_compute_col_sums_ = self.get_kernel("compute_col_sums")
            self.krn_compute_col_sums_.set_args(self.weights.devmem,
                                                self.col_sums.devmem)
            self.krn_weights_.set_arg(13, self.col_sums.devmem)

        self.assign_kernel("Unpack1D")
        unpack_bytes = (self._kernel_app_per_image * self.unpack_size *
                        self._kernel_size * self.err_output.itemsize)
        self.device.request_temp_buffer(unpack_bytes)

        if self.need_err_input:
            self.krn_err_input_ = self.get_kernel("DirectPack")
            self.krn_err_input_scale_ = self.get_kernel("Scale")
            self.krn_err_input_scale_.set_arg(0, self.err_input.devmem)
            self.np_err_input_alpha = numpy.ones(1, dtype=self._dtype)
            self.np_err_input_beta = numpy.zeros(1, dtype=self._dtype)

        self.gemm_ = blas_class.gemm(self._dtype)
        self.np_one = numpy.ones(1, dtype=self._dtype)
        self.np_zero = numpy.zeros(1, dtype=self._dtype)
        self._const_i = numpy.zeros(2, dtype=numpy.int64)

    def ocl_init(self):
        ocl_blas.OCLBLAS.attach_to_device(self.device)
        self._gpu_init(ocl_blas.OCLBLAS)

        if self.need_gradient_weights:
            self._global_size_weights = (self.weights.size,)
            self._local_size_weights = None

        if self.need_gradient_weights and self.include_bias:
            self._global_size_bias = (self._side * self.reduce_size,)
            self._local_size_bias = (self.reduce_size,)

        if self.need_gradient_weights:
            self._global_size_ortho = (self._other * self.reduce_size,)
            self._local_size_ortho = (self.reduce_size,)

        self._global_size_unpack = lambda size: (size,)
        self._local_size_unpack = None

        if self.need_err_input:
            self.krn_err_input_clear_ = self.get_kernel("err_input_clear")
            self.krn_err_input_clear_.set_arg(0, self.err_input.devmem)
            self._err_input_clear = (
                lambda: self.execute_kernel(
                    (self.err_input.size,), None, self.krn_err_input_clear_))

            self._global_size_err_input = lambda size: (size,)
            self._local_size_err_input = None

            self.krn_err_input_.set_arg(1, self.err_input.devmem)

            self._global_size_err_input_scale = (self.err_input.size,)
            self._local_size_err_input_scale = None

        self._process_err_input_subblock = (
            self._ocl_process_err_input_subblock)
        self._process_weights_subblock = (
            self._ocl_process_weights_subblock)

        self.set_arg(0, self.input)

    def cuda_init(self):
        self._gpu_init(cublas.CUBLAS)

        if self.need_gradient_weights:
            block_size = self.device.suggest_block_size(self.krn_weights_)
            self._global_size_weights = (int(numpy.ceil(
                self.weights.size / block_size)), 1, 1)
            self._local_size_weights = (block_size, 1, 1)

        if self.include_bias:
            self._global_size_bias = (self._side, 1, 1)
            self._local_size_bias = (self.reduce_size, 1, 1)

        if self.need_gradient_weights:
            self._global_size_ortho = (self._other, 1, 1)
            self._local_size_ortho = (self.reduce_size, 1, 1)

        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size_unpack = (
            lambda size: (int(numpy.ceil(size / block_size)), 1, 1))
        self._local_size_unpack = (block_size, 1, 1)

        if self.need_err_input:
            self._err_input_clear = (
                lambda: self.err_input.devmem.memset32_async())
            block_size = self.device.suggest_block_size(self.krn_err_input_)
            self._global_size_err_input = (
                lambda size: (int(numpy.ceil(size / block_size)), 1, 1))
            self._local_size_err_input = (block_size, 1, 1)

            block_size = self.device.suggest_block_size(
                self.krn_err_input_scale_)
            self._global_size_err_input_scale = (
                int(numpy.ceil(self.err_input.size / block_size)), 1, 1)
            self._local_size_err_input_scale = (block_size, 1, 1)
            self.krn_err_input_scale_.set_arg(2, self.err_input.size)

        self._process_err_input_subblock = (
            self._cuda_process_err_input_subblock)
        self._process_weights_subblock = (
            self._cuda_process_weights_subblock)

    def gpu_err_input_update(self):
        if not self.need_err_input:
            return

        self.unmap_vectors(self.err_input, self.err_output, self.weights)
        unpack_data = self.device.get_temp_buffer()

        if not self.err_input_beta:
            self._err_input_clear()
        else:
            self.np_err_input_beta[0] = self.err_input_beta
            self.krn_err_input_scale_.set_arg(1, self.np_err_input_beta)
            self.execute_kernel(
                self._global_size_err_input_scale,
                self._local_size_err_input_scale, self.krn_err_input_scale_)

        for i in range(0, self._batch_size, self.unpack_size):
            self._process_err_input_subblock(
                i, min(self._batch_size - i, self.unpack_size), unpack_data)

    def _cuda_process_err_input_subblock(self, start_image, image_count,
                                         unpack_data):
        output_offs = (start_image * self.err_output.sample_size *
                       self.err_output.itemsize)
        unpack_side = self._kernel_app_per_image * image_count

        self.np_err_input_alpha[0] = self.err_input_alpha
        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_T if self.weights_transposed
            else cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
            self._kernel_size, unpack_side, self.weights_shape[0],
            self.np_err_input_alpha, self.weights.devmem,
            int(self.err_output.devmem) + output_offs,
            self.np_zero, unpack_data)

        self.krn_err_input_.set_arg(0, unpack_data)
        self.krn_err_input_.set_arg(
            1, int(self.err_input.devmem) +
            start_image * self.input.sample_size * self.input.itemsize)
        limit = unpack_side * self._kernel_size
        self._const_i[0] = limit
        self.krn_err_input_.set_arg(2, self._const_i[0:1])
        self.execute_kernel(self._global_size_err_input(limit),
                            self._local_size_err_input, self.krn_err_input_)

    def _ocl_process_err_input_subblock(self, start_image, image_count,
                                        unpack_data):
        output_offs = start_image * self.err_output.sample_size
        unpack_side = self._kernel_app_per_image * image_count

        self.np_err_input_alpha[0] = self.err_input_alpha
        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_T if self.weights_transposed
            else cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
            self._kernel_size, unpack_side, self.weights_shape[0],
            self.np_err_input_alpha, self.weights.devmem,
            self.err_output.devmem,
            self.np_zero, unpack_data, offsetB=output_offs)

        self.krn_err_input_.set_arg(0, unpack_data)
        self._const_i[0] = start_image * self.input.sample_size
        self.krn_err_input_.set_arg(2, self._const_i[0:1])
        limit = unpack_side * self._kernel_size
        self.execute_kernel(self._global_size_err_input(limit),
                            self._local_size_err_input, self.krn_err_input_)

    def gpu_weights_update(self):
        if not self.need_gradient_weights:
            return
        self.unmap_vectors(self.err_output, self.input, self.gradient_weights)
        unpack_data = self.device.get_temp_buffer()

        # Calculate weights gradient: err_output * input
        for i in range(0, self._batch_size, self.unpack_size):
            self._process_weights_subblock(
                i, min(self._batch_size - i, self.unpack_size), unpack_data)

        # Apply learning_rate etc.
        super(GradientDescentConv, self).gpu_weights_update()

    def _cuda_process_weights_subblock(self, start_image, image_count,
                                       unpack_data):
        # Unpack
        self._kernel_.set_arg(
            0, int(self.input.devmem) +
            start_image * self.input.sample_size * self.input.itemsize)
        self._kernel_.set_arg(1, unpack_data)
        unpack_side = self._kernel_app_per_image * image_count
        limit = unpack_side * self._kernel_size
        self._const_i[1] = limit
        self._kernel_.set_arg(2, self._const_i[1:2])
        self.execute_kernel(self._global_size_unpack(limit),
                            self._local_size_unpack)
        output_offs = (start_image * self.err_output.sample_size *
                       self.err_output.itemsize)

        # Accumulate gradient
        if self.weights_transposed:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.n_kernels, self._kernel_size, unpack_side,
                self.np_one, int(self.err_output.devmem) + output_offs,
                unpack_data, self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem)
        else:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self._kernel_size, self.n_kernels, unpack_side, self.np_one,
                unpack_data, int(self.err_output.devmem) + output_offs,
                self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem)

    def _ocl_process_weights_subblock(self, start_image, image_count,
                                      unpack_data):
        # Unpack
        self._const_i[1] = start_image * self.input.sample_size
        self._kernel_.set_arg(1, unpack_data)
        self._kernel_.set_arg(2, self._const_i[1:2])
        unpack_side = self._kernel_app_per_image * image_count
        limit = unpack_side * self._kernel_size
        self.execute_kernel(self._global_size_unpack(limit),
                            self._local_size_unpack)
        output_offs = start_image * self.err_output.sample_size

        # Accumulate gradient
        if self.weights_transposed:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.n_kernels, self._kernel_size, unpack_side,
                self.np_one, self.err_output.devmem,
                unpack_data, self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem, offsetA=output_offs)
        else:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self._kernel_size, self.n_kernels, unpack_side, self.np_one,
                unpack_data, self.err_output.devmem,
                self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem, offsetB=output_offs)

    def numpy_weights_update(self):
        if not self.need_gradient_weights:
            return
        self.input.map_read()
        self.err_output.map_read()
        self.weights.map_write()
        self.gradient_weights.map_write()
        self.accumulated_gradient_weights.map_write()

        dtype = self.weights.dtype
        sy = self.input.shape[1]
        sx = self.input.shape[2]
        n_channels = self.input.size // (self.input.shape[0] * sx * sy)

        sx_full = self.padding[0] + sx + self.padding[2]
        sy_full = self.padding[1] + sy + self.padding[3]
        nx = (sx_full - self.kx) // self.sliding[0] + 1
        ny = (sy_full - self.ky) // self.sliding[1] + 1
        sample_shape = (nx * ny, self.kx * self.ky * n_channels)

        sh = self.err_output.shape
        if len(sh) == 3:
            sh[1] *= sh[2]
            sh[2] = 1

        # calculate gradient for weights
        gd_weights = (reshape_transposed(self.gradient_weights.mem)
                      if self.weights_transposed
                      else self.gradient_weights.mem)
        gd_weights[:] = 0
        cut = numpy.empty((self.ky, self.kx, n_channels), dtype=dtype)
        sample = numpy.empty(sample_shape, dtype=dtype)
        for batch in range(self.current_batch_size):
            # input data unrolling
            sample = numpy.empty(sample_shape)
            for by, bx in ((by, bx) for by in range(ny) for bx in range(nx)):
                y1, y2 = (by * self.sliding[1],
                          by * self.sliding[1] + self.ky)
                x1, x2 = (bx * self.sliding[0],
                          bx * self.sliding[0] + self.kx)
                i1, i2 = (min(max(y1 - self.padding[1], 0), sy),
                          min(max(y2 - self.padding[1], 0), sy))
                j1, j2 = (min(max(x1 - self.padding[0], 0), sx),
                          min(max(x2 - self.padding[0], 0), sx))
                cut_i1, cut_i2 = (i1 - y1 + self.padding[1],
                                  i2 - y1 + self.padding[1])
                cut_j1, cut_j2 = (j1 - x1 + self.padding[0],
                                  j2 - x1 + self.padding[0])
                cut = numpy.zeros((self.ky, self.kx, n_channels),
                                  dtype=self.input.mem.dtype)
                cut[cut_i1:cut_i2, cut_j1:cut_j2, :] = \
                    self.input.mem[batch, i1:i2, j1:j2, :].reshape(i2 - i1,
                                                                   j2 - j1,
                                                                   n_channels)
                sample[by * nx + bx] = cut.ravel()
            err_out_shape = self.err_output.mem.shape
            out = self.err_output.mem[batch].reshape(err_out_shape[1] *
                                                     err_out_shape[2],
                                                     self.n_kernels)
            gd_weights += numpy.dot(out.transpose(),
                                    sample)
        if self.weights_transposed:
            gd_weights = reshape_transposed(gd_weights)

        # update weights
        lr = self.learning_rate
        factor_l12 = self.weights_decay
        l1_vs_l2 = self.l1_vs_l2
        gradient = -nn_units.GradientDescentBase.numpy_gradient_step(
            self.weights.mem, gd_weights, lr, factor_l12, l1_vs_l2,
            self.factor_ortho, self.weights_transposed)

        if self.accumulate_gradient:
            self.accumulate_gradient_f(self.accumulated_gradient_weights.mem,
                                       gradient)

        if self.gradient_weights_with_moment:
            gradient += (self.gradient_weights_with_moment.mem *
                         self.gradient_moment)
            self.gradient_weights.mem[:] = gradient[:]
        if self.apply_gradient:
            self.weights.mem += gradient

    def numpy_bias_update(self):
        if not self.need_gradient_weights or not self.include_bias:
            return

        self.err_output.map_read()
        self.bias.map_write()
        self.gradient_bias.map_write()
        self.accumulated_gradient_bias.map_write()

        err_out_shape = self.err_output.mem.shape

        # calculate gradient for bias
        gd_bias = self.gradient_bias.mem
        gd_bias[:] = 0
        for batch in range(self.current_batch_size):
            out = self.err_output.mem[batch].reshape(err_out_shape[1] *
                                                     err_out_shape[2],
                                                     self.n_kernels)
            gd_bias += numpy.add.reduce(out)
        # update bias
        lr = self.learning_rate_bias
        factor_l12 = self.weights_decay_bias
        l1_vs_l2 = self.l1_vs_l2_bias

        gd_bias_reg = -nn_units.GradientDescentBase.numpy_gradient_step(
            self.bias.mem, gd_bias, lr, factor_l12, l1_vs_l2)

        if self.accumulate_gradient:
            self.accumulate_gradient_f(self.accumulated_gradient_bias.mem,
                                       gd_bias_reg)

        if self.gradient_bias_with_moment:
            gd_bias_reg += (self.gradient_bias_with_moment.mem *
                            self.gradient_moment_bias)
            self.gradient_bias_with_moment.mem[:] = gd_bias_reg[:]
        if self.apply_gradient:
            self.bias.mem += gd_bias_reg

    def numpy_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return

        from scipy.signal import convolve2d

        self.err_input.map_invalidate()
        self.err_output.map_read()
        self.weights.map_read()

        batch_size = self.input.mem.shape[0]
        sy = self.input.mem.shape[1]
        sx = self.input.mem.shape[2]
        n_channels = self.input.mem.size // (batch_size * sx * sy)
        sx_full = self.padding[0] + sx + self.padding[2]
        sy_full = self.padding[1] + sy + self.padding[3]

        weights = (reshape_transposed(self.weights.mem)
                   if self.weights_transposed else self.weights.mem)

        if not self.err_input_beta:
            self.err_input.mem[:] = 0
        else:
            self.err_input.mem *= self.err_input_beta
        err_input = numpy.zeros_like(self.err_input.mem)
        # initialize sparse output error
        sparse_err_output = numpy.zeros((
            batch_size, sy_full - self.ky + 1, sx_full - self.kx + 1,
            self.n_kernels), dtype=self.err_output.dtype)
        for (batch, i, j, k), err in numpy.ndenumerate(self.err_output.mem):
            sparse_err_output[batch, i * self.sliding[1],
                              j * self.sliding[0], k] = err
        err_sample = numpy.zeros(
            (sy_full - self.ky + 1, sx_full - self.kx + 1),
            dtype=err_input.dtype)
        for batch, k in product(range(batch_size), range(self.n_kernels)):
            err_sample[:] = sparse_err_output[batch, :, :, k]
            cur_kernel = weights[k].reshape(self.ky, self.kx, n_channels)
            for ch in range(n_channels):
                err_input_full = convolve2d(err_sample, cur_kernel[:, :, ch],
                                            mode='full')
                err_input[batch, :, :, ch] += \
                    err_input_full[self.padding[1]:(sy_full - self.padding[3]),
                                   self.padding[0]:(sx_full - self.padding[2])]
        self.err_input.mem += err_input * self.err_input_alpha

    def gpu_run(self):
        """Do gradient descent for OpenCL and CUDA.
        """
        self.gpu_err_output_update()
        self.gpu_err_input_update()
        self.gpu_weights_update()
        self.gpu_bias_update()
        self.print_debug_data()

    def ocl_run(self):
        self.gpu_run()

    def cuda_run(self):
        self.gpu_run()

    def numpy_run(self):
        self.numpy_err_output_update()
        self.numpy_err_input_update()
        self.numpy_weights_update()
        self.numpy_bias_update()
        self.print_debug_data()


class GDTanhConv(nn_units.GradientDescentWithActivation, GradientDescentConv):
    """Gradient Descent for f(x) = 1.7159 * tanh(0.6666 * s), s = (W * x + b),
       y = a * tanh(b * s).

    f'(s) = (a * tanh(b * s))' = a * tanh'(b * s) * b
          = a * (1.0 - tanh^2(b * s)) * b
          = a * b - a * b * tanh^2(b * s)
          = a * b - y * y * b / a
          = y * y * (-b / a) + (a * b)
          = y * y * (-0.388484177) + 1.14381894
    """

    MAPPING = {"conv_tanh"}

    def numpy_err_output_update(self):
        """Multiply err_output by activation derivative by s
           in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= output * output * (-0.388484177) + 1.14381894

    def initialize(self, device, **kwargs):
        self.sources_["gradient_descent_tanh"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        super(GDTanhConv, self).initialize(device=device, **kwargs)


class GDSigmoidConv(nn_units.GradientDescentWithActivation,
                    GradientDescentConv):
    """Gradient Descent for f(x) = 1.0 / (1.0 + exp(-s)), s = (W * x + b),
       y = 1.0 / (1.0 + exp(-s)).

    f'(s) = y * (1 - y).
    """

    MAPPING = {"conv_sigmoid"}

    def numpy_err_output_update(self):
        """Multiply err_output by activation derivative by s
           in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= output * (1.0 - output)

    def initialize(self, device, **kwargs):
        self.sources_["gradient_descent_sigmoid"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        super(GDSigmoidConv, self).initialize(device=device, **kwargs)


class GDRELUConv(nn_units.GradientDescentWithActivation, GradientDescentConv):
    """Gradient Descent for f(x) = log(1.0 + exp(s)), s = (W * x + b),
       y = log(1.0 + exp(s)).

    f'(s) = 1.0 / (1.0 + exp(-s)) = 1.0 - exp(-y)
    """

    MAPPING = {"conv_relu"}

    def numpy_err_output_update(self):
        """Multiply err_output by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= 1.0 - numpy.exp(-output)

    def initialize(self, device, **kwargs):
        self.sources_["gradient_descent_relu"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        super(GDRELUConv, self).initialize(device=device, **kwargs)


class GDStrictRELUConv(nn_units.GradientDescentWithActivation,
                       GradientDescentConv):
    """Gradient Descent for strict ReLU (like in CAFFE)

    :math:`f(x) = \\max(x, 0)`

    :math:`f'(s) = \\begin{cases}1 & s > 0 \\\\ 0 & else. \\\\ \\end{cases}`
    """

    MAPPING = {"conv_str"}

    def numpy_err_output_update(self):
        """Multiply `err_output` by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= numpy.greater(output, 0)

    def initialize(self, device, **kwargs):
        self.sources_["gradient_descent_strict_relu"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        super(GDStrictRELUConv, self).initialize(device=device, **kwargs)
