"""
Created on Nov 14, 2013

Gradient descent for convolutional units.

* :class:`GradientDescentConv` couples with :class:`veles.znicz.conv.Conv`
* :class:`GDTanhConv` couples with :class:`veles.znicz.conv.ConvTanh`
* :class:`GDRELUConv` couples with :class:`veles.znicz.conv.ConvRELU`
* :class:`GDStrictRELUConv` couples with \
    :class:`veles.znicz.conv.ConvStrictRELU`

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import cuda4py.blas as cublas
from itertools import product
import numpy
import time
from zope.interface import implementer

import veles.error as error
from veles.memory import reshape_transposed, roundup
from veles.accelerated_units import IOpenCLUnit, ICUDAUnit
from veles.znicz.conv import ConvolutionalBase
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit, ICUDAUnit)
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
        self.reduce_size = 64
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

    def _gpu_init(self):
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

        self.krn_weights_ = self.get_kernel("weights_update")
        self.krn_weights_.set_args(self.err_output.devmem,
                                   self.input.devmem,
                                   self.weights.devmem,
                                   self.gradient_weights.devmem,
                                   self.accumulated_gradient_weights.devmem,
                                   self.gradient_weights_with_moment.devmem)

        if self.include_bias:
            self.krn_bias_ = self.get_kernel("bias_update")
            self.krn_bias_.set_args(
                self.err_output.devmem, self.bias.devmem,
                self.gradient_bias.devmem,
                self.accumulated_gradient_bias.devmem,
                self.gradient_bias_with_moment.devmem)

        if self.factor_ortho:
            self.krn_compute_col_sums_ = self.get_kernel("compute_col_sums")
            self.krn_compute_col_sums_.set_args(self.weights.devmem,
                                                self.col_sums.devmem)
            self.krn_weights_.set_arg(11, self.col_sums.devmem)

    def ocl_init(self):
        a_width = self._kernel_app_total
        b_width = self._kernel_size
        block_size = self.device.device_info.get_block_size(
            kernel="deconv", dtype=self.err_output.dtype)
        self.sources_["conv/gradient_descent/err_input_update"] = {
            "BLOCK_SIZE": block_size
        }
        self._global_size_err_input = [
            roundup(b_width, block_size),
            roundup(a_width, block_size)]
        self._local_size_err_input = [block_size, block_size]

        a_width = (self._kernel_size if self.weights_transposed
                   else self.n_kernels)
        b_width = (self.n_kernels if self.weights_transposed
                   else self._kernel_size)
        block_size = self.device.device_info.get_block_size(
            kernel="conv", dtype=self.err_output.dtype)
        self.sources_["conv/gradient_descent/weights_update"] = {
            "BLOCK_SIZE": block_size,
            "USE_ORTHO": int(bool(self.factor_ortho)),
            "USE_MOMENT": int(bool(self.gradient_moment))
        }
        self._global_size_weights = [
            roundup(b_width, block_size),
            roundup(a_width, block_size)]
        self._local_size_weights = [block_size, block_size]

        self.sources_["conv/gradient_descent/bias_update"] = {
            "USE_MOMENT": int(bool(self.gradient_moment_bias))
        }
        self._global_size_bias = [self.n_kernels * self.reduce_size]
        self._local_size_bias = [self.reduce_size]

        self._gpu_init()

        self._global_size_ortho = [self._other * self.reduce_size]
        self._local_size_ortho = [self.reduce_size]

        if self.need_err_input:
            self.krn_err_input_clear_ = self.get_kernel("err_input_clear")
            self.krn_err_input_clear_.set_arg(0, self.err_input.devmem)

            self.krn_err_input_ = self.get_kernel("err_input_update")
            self.krn_err_input_.set_args(self.err_output.devmem,
                                         self.weights.devmem,
                                         self.err_input.devmem)

    def cuda_init(self):
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

        self._gpu_init()

        block_size = self.device.suggest_block_size(self.krn_weights_)
        self._global_size_weights = (int(numpy.ceil(
            self.weights.size / block_size)), 1, 1)
        self._local_size_weights = (block_size, 1, 1)

        if self.include_bias:
            self._global_size_bias = (self._side, 1, 1)
            self._local_size_bias = (self.reduce_size, 1, 1)

        self._global_size_ortho = (self._other, 1, 1)
        self._local_size_ortho = (self.reduce_size, 1, 1)

        unpack_shape = (self._kernel_app_per_image * self.unpack_size,
                        self._kernel_size)
        if not self.unpack_data:
            self.unpack_data.reset(numpy.zeros(unpack_shape, self._dtype))
        else:
            assert self.unpack_data.shape == unpack_shape
        self.unpack_data.initialize(self.device)

        self.assign_kernel("Unpack1D")
        self.set_arg(1, self.unpack_data)
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size_unpack = (
            lambda size: (int(numpy.ceil(size / block_size)), 1, 1))
        self._local_size_unpack = (block_size, 1, 1)

        if self.need_err_input:
            self.krn_err_input_ = self.get_kernel("DirectPack")
            block_size = self.device.suggest_block_size(self.krn_err_input_)
            self._global_size_err_input = (
                lambda size: (int(numpy.ceil(size / block_size)), 1, 1))
            self._local_size_err_input = (block_size, 1, 1)
            self.krn_err_input_.set_arg(0, self.unpack_data.devmem)

        self.gemm_ = cublas.CUBLAS.gemm(self._dtype)
        self.np_one = numpy.ones(1, dtype=self._dtype)
        self.np_zero = numpy.zeros(1, dtype=self._dtype)
        self._const_i = numpy.zeros(2, dtype=numpy.int64)

    def cuda_err_input_update(self):
        if not self.need_err_input:
            return

        self.unmap_vectors(self.err_input, self.err_output, self.weights,
                           self.unpack_data)

        self.err_input.devmem.memset32_async()
        for i in range(0, self._batch_size, self.unpack_size):
            self._process_err_input_subblock(i, min(self._batch_size - i,
                                                    self.unpack_size))

    def _process_err_input_subblock(self, start_image, image_count):
        output_offs = (start_image * self.err_output.sample_size *
                       self.err_output.itemsize)
        unpack_side = self._kernel_app_per_image * image_count

        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_T if self.weights_transposed
            else cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
            self._kernel_size, unpack_side, self.weights_shape[0],
            self.np_one, self.weights.devmem,
            int(self.err_output.devmem) + output_offs,
            self.np_zero, self.unpack_data.devmem)

        self.krn_err_input_.set_arg(1, int(self.err_input.devmem) +
                                    start_image * self.input.sample_size *
                                    self.input.itemsize)
        limit = unpack_side * self._kernel_size
        self._const_i[0] = limit
        self.krn_err_input_.set_arg(2, self._const_i[0:1])
        self.execute_kernel(self._global_size_err_input(limit),
                            self._local_size_err_input, self.krn_err_input_)

    def cuda_weights_update(self):
        self.unmap_vectors(self.err_output, self.input, self.gradient_weights,
                           self.weights, self.accumulated_gradient_weights)

        # Calculate weights gradient: err_output * input
        for i in range(0, self._batch_size, self.unpack_size):
            self._process_weights_subblock(i, min(self._batch_size - i,
                                                  self.unpack_size))

        # Apply learning_rate etc.
        self.gpu_weights_update()

    def _process_weights_subblock(self, start_image, image_count):
        # Unpack
        self._kernel_.set_arg(0, int(self.input.devmem) +
                              start_image * self.input.sample_size *
                              self.input.itemsize)
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
                self.unpack_data.devmem,
                self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem)
        else:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self._kernel_size, self.n_kernels, unpack_side,
                self.np_one, self.unpack_data.devmem,
                int(self.err_output.devmem) + output_offs,
                self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem)

    def cpu_weights_update(self):
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
        gradient = -nn_units.GradientDescentBase.cpu_gradient_step(
            self.weights.mem, gd_weights, lr, factor_l12, l1_vs_l2,
            self.factor_ortho, self.weights_transposed)
        if self.accumulate_gradient == self.OP_NONE:
            pass
        elif self.accumulate_gradient == self.OP_STORE:
            self.accumulated_gradient_weights.mem[:] = gradient
        elif self.accumulate_gradient == self.OP_ADD:
            self.accumulated_gradient_weights.mem[:] += gradient
        elif self.accumulate_gradient == self.OP_FLUSH:
            gradient += self.accumulated_gradient_weights.mem
            self.accumulated_gradient_weights.mem[:] = 0
        else:
            raise ValueError("Incorrect accumulate_gradient attribute value")
        if self.gradient_weights_with_moment:
            gradient += (self.gradient_weights_with_moment.mem *
                         self.gradient_moment)
            self.gradient_weights.mem[:] = gradient[:]
        if self.apply_gradient:
            self.weights.mem += gradient

    def cpu_bias_update(self):
        if not self.include_bias:
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
        lr = self.learning_rate
        factor_l12 = self.weights_decay
        l1_vs_l2 = self.l1_vs_l2

        gd_bias_reg = -nn_units.GradientDescentBase.cpu_gradient_step(
            self.bias.mem, gd_bias, lr, factor_l12, l1_vs_l2)

        if self.accumulate_gradient == self.OP_NONE:
            pass
        elif self.accumulate_gradient == self.OP_STORE:
            self.accumulated_gradient_bias.mem[:] = gd_bias_reg
        elif self.accumulate_gradient == self.OP_ADD:
            self.accumulated_gradient_bias.mem[:] += gd_bias_reg
        elif self.accumulate_gradient == self.OP_FLUSH:
            gd_bias_reg += self.accumulated_gradient_bias.mem
            self.accumulated_gradient_bias.mem[:] = 0
        else:
            raise ValueError("Incorrect accumulate_gradient attribute value")

        if self.gradient_bias_with_moment:
            gd_bias_reg += (self.gradient_bias_with_moment.mem *
                            self.gradient_moment_bias)
            self.gradient_bias_with_moment.mem[:] = gd_bias_reg[:]
        if self.apply_gradient:
            self.bias.mem += gd_bias_reg

    def ocl_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return

        self.unmap_vectors(self.err_input, self.err_output, self.weights)
        # Clear the resulting matrix
        self.execute_kernel([self.err_input.mem.size], None,
                            self.krn_err_input_clear_)

        self.execute_kernel(
            self._global_size_err_input, self._local_size_err_input,
            self.krn_err_input_)

    def cpu_err_input_update(self):
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

        self.err_input.mem[:] = 0
        # initialize sparse output error
        sparse_err_output = numpy.zeros((
            batch_size, sy_full - self.ky + 1, sx_full - self.kx + 1,
            self.n_kernels), dtype=self.err_output.dtype)
        for (batch, i, j, k), err in numpy.ndenumerate(self.err_output.mem):
            sparse_err_output[batch, i * self.sliding[1],
                              j * self.sliding[0], k] = err
        err_sample = numpy.empty((sy_full - self.ky + 1,
                                  sx_full - self.kx + 1))
        for batch, k in product(range(batch_size), range(self.n_kernels)):
            err_sample[:] = sparse_err_output[batch, :, :, k]
            cur_kernel = weights[k].reshape(self.ky, self.kx, n_channels)
            for ch in range(n_channels):
                err_input_full = convolve2d(err_sample, cur_kernel[:, :, ch],
                                            mode='full')
                self.err_input.mem[batch, :, :, ch] += \
                    err_input_full[self.padding[1]:(sy_full - self.padding[3]),
                                   self.padding[0]:(sx_full - self.padding[2])]

    def ocl_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_output_update()
        self.ocl_err_input_update()
        self.gpu_weights_update()
        self.gpu_bias_update()
        self.print_debug_data(t1)

    def cuda_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_output_update()
        self.cuda_err_input_update()
        self.cuda_weights_update()
        self.gpu_bias_update()
        self.print_debug_data(t1)

    def cpu_run(self):
        t1 = time.time()
        self.cpu_err_output_update()
        self.cpu_err_input_update()
        self.cpu_weights_update()
        self.cpu_bias_update()
        self.print_debug_data(t1)


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

    def cpu_err_output_update(self):
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

    def cpu_err_output_update(self):
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

    def cpu_err_output_update(self):
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

    def cpu_err_output_update(self):
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
