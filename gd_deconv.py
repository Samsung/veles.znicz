"""
Created on Jul 2, 2014

Gradient descent for deconvolutional layer.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import cuda4py.blas as cublas
import numpy
from zope.interface import implementer
from veles.compat import from_none

from veles.memory import roundup
from veles.accelerated_units import IOpenCLUnit, ICUDAUnit
import veles.znicz.nn_units as nn_units
from veles.znicz.conv import ConvolutionalBase
from veles.znicz.deconv import Deconv


@implementer(IOpenCLUnit, ICUDAUnit)
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
        self.reduce_size = 64
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
            padding = Deconv.compute_padding(
                sx, sy, self.kx, self.ky, self.sliding)
            if self.padding is None:  # pylint: disable=E0203
                self.padding = padding
            elif self.padding != padding:
                raise ValueError(
                    "Expected padding %s got %s"
                    % (str(padding), str(self.padding)))
        except ValueError as e:
            if not self.hits:
                raise from_none(e)
            self.warning("Using unsafe padding of %s", self.padding)

        if self.hits:
            self.hits.initialize(self.device)

        self._dtype = self.err_output.dtype

    def _gpu_init(self, defines):
        self._batch_size = self.err_output.shape[0]
        sy, sx = self.ky_kx
        self._kernel_app_per_image = self.input.sample_size // self.n_kernels
        self._kernel_app_total = (self._kernel_app_per_image *
                                  self.input.shape[0])
        self._kernel_size = self.kx * self.ky * self.channels_number

        self.cl_const = numpy.zeros(5, dtype=self._dtype)

        side = self.weights_shape[0]
        other = self.weights.size // side
        assert side == self.n_kernels
        assert other == self.kx * self.ky * self.channels_number

        defines.update({
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
        })

        self.build_program(
            defines, "%s_%d_%d_%d" % (
                self.__class__.__name__,
                self.input.shape[0],
                self.input.sample_size,
                self.err_output.sample_size),
            dtype=self._dtype)

        self.krn_err_output_ = self.get_kernel("err_output_update")
        self.krn_err_output_.set_arg(0, self.err_output.devmem)
        if self.hits:
            self.krn_err_output_.set_arg(1, self.hits.devmem)

        self.krn_weights_ = self.get_kernel("weights_update")
        self.krn_weights_.set_args(self.err_output.devmem,
                                   self.input.devmem,
                                   self.weights.devmem,
                                   self.gradient_weights.devmem,
                                   self.accumulated_gradient_weights.devmem,
                                   self.gradient_weights_with_moment.devmem)

        if self.factor_ortho:
            self.krn_compute_col_sums_ = self.get_kernel("compute_col_sums")
            self.krn_compute_col_sums_.set_args(self.weights.devmem,
                                                self.col_sums.devmem)
            self.krn_weights_.set_arg(11, self.col_sums.devmem)

    def ocl_init(self):
        block_size_err_input = self.device.device_info.get_block_size(
            kernel="conv", dtype=self._dtype)
        self.sources_["conv/forward"] = {
            "BLOCK_SIZE": block_size_err_input,
        }
        block_size_weights = self.device.device_info.get_block_size(
            kernel="conv", dtype=self._dtype)
        self.sources_["deconv/gradient_descent/weights_update"] = {
            "BLOCK_SIZE": block_size_weights,
            "USE_ORTHO": int(bool(self.factor_ortho)),
            'USE_MOMENT': int(bool(self.gradient_moment))
        }

        self._gpu_init({})

        self._global_size_err_output = (self.err_output.size,)
        self._local_size_err_output = None

        a_width = self._kernel_app_total
        b_width = self.n_kernels
        self._global_size_err_input = [
            roundup(b_width, block_size_err_input),
            roundup(a_width, block_size_err_input)]
        self._local_size_err_input = [block_size_err_input,
                                      block_size_err_input]

        a_width = (self._kernel_size if self.weights_transposed
                   else self.n_kernels)
        b_width = (self.n_kernels if self.weights_transposed
                   else self._kernel_size)
        self._global_size_weights = [
            roundup(b_width, block_size_weights),
            roundup(a_width, block_size_weights)]
        self._local_size_weights = [block_size_weights, block_size_weights]

        if self.need_err_input:
            self.krn_err_input_ = self.get_kernel("feed_layer")
            self.krn_err_input_.set_args(self.err_output.devmem,
                                         self.weights.devmem,
                                         self.err_input.devmem)

    def cuda_init(self):
        self.sources_["conv/forward"] = {}
        self.sources_["deconv/gradient_descent/weights_update"] = {
            "USE_ORTHO": int(bool(self.factor_ortho)),
            'USE_MOMENT': int(bool(self.gradient_moment))
        }
        self._gpu_init({})

        block_size = self.device.suggest_block_size(self.krn_err_output_)
        self._global_size_err_output = (
            int(numpy.ceil(self.err_output.size / block_size)), 1, 1)
        self._local_size_err_output = (block_size, 1, 1)

        self.gemm_ = cublas.CUBLAS.gemm(self._dtype)
        self.np_one = numpy.ones(1, dtype=self._dtype)
        self.np_zero = numpy.zeros(1, dtype=self._dtype)
        self._const_i = numpy.zeros(1, dtype=numpy.int64)

        self.assign_kernel("Unpack1D")

        unpack_shape = (self._kernel_app_per_image * self.unpack_size,
                        self._kernel_size)
        if not self.unpack_data:
            self.unpack_data.reset(numpy.zeros(unpack_shape, self._dtype))
        else:
            assert self.unpack_data.shape == unpack_shape

        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size_unpack = (
            lambda size: (int(numpy.ceil(size / block_size)), 1, 1))
        self._local_size_unpack = (block_size, 1, 1)

        self.unpack_data.initialize(self.device)

        self.set_arg(1, self.unpack_data)

        block_size = self.device.suggest_block_size(self.krn_weights_)
        self._global_size_weights = (
            int(numpy.ceil(self.weights.size / block_size)), 1, 1)
        self._local_size_weights = (block_size, 1, 1)

    def gpu_err_output_update(self):
        self.err_output.unmap()
        self.execute_kernel(
            self._global_size_err_output, self._local_size_err_output,
            self.krn_err_output_)

    def ocl_err_input_update(self):
        if not self.need_err_input:
            return
        self.unmap_vectors(self.err_input, self.err_output, self.weights)

        self.execute_kernel(
            self._global_size_err_input, self._local_size_err_input,
            self.krn_err_input_)

    def ocl_run(self):
        self.gpu_err_output_update()
        self.ocl_err_input_update()
        self.gpu_weights_update()

    def cuda_run(self):
        # Divide err_output by hits count
        self.gpu_err_output_update()

        # Update err_input and simultaneousely accumulate gradient
        self.unmap_vectors(self.err_input, self.weights, self.err_output,
                           self.unpack_data, self.gradient_weights)
        for i in range(0, self._batch_size, self.unpack_size):
            self._process_subblock(i, min(self._batch_size - i,
                                          self.unpack_size))

        # Update weights
        self.gpu_weights_update()

    def _process_subblock(self, start_image, image_count):
        self._kernel_.set_arg(0, int(self.err_output.devmem) +
                              start_image * self.err_output.sample_size *
                              self.err_output.itemsize)
        unpack_side = self._kernel_app_per_image * image_count
        limit = unpack_side * self._kernel_size
        self._const_i[0] = limit
        self._kernel_.set_arg(2, self._const_i)
        self.execute_kernel(self._global_size_unpack(limit),
                            self._local_size_unpack)
        output_offs = (start_image * self.err_input.sample_size *
                       self.err_input.itemsize)

        # Update err_input
        if self.need_err_input:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N if self.weights_transposed
                else cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
                self.weights_shape[0], unpack_side, self._kernel_size,
                self.np_one, self.weights.devmem, self.unpack_data.devmem,
                self.np_zero, int(self.err_input.devmem) + output_offs)

        # Accumulate gradient
        if self.weights_transposed:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.n_kernels, self._kernel_size, unpack_side,
                self.np_one, int(self.input.devmem) + output_offs,
                self.unpack_data.devmem,
                self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem)
        else:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self._kernel_size, self.n_kernels, unpack_side,
                self.np_one, self.unpack_data.devmem,
                int(self.input.devmem) + output_offs,
                self.np_one if start_image else self.np_zero,
                self.gradient_weights.devmem)

    def cpu_run(self):
        raise NotImplementedError()
