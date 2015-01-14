"""
Created on Jul 2, 2014

Gradient descent for deconvolutional layer.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import numpy
from zope.interface import implementer

import veles.error as error
from veles.memory import roundup
from veles.accelerated_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units
from veles.znicz.conv import ConvolutionalBase
from veles.znicz.deconv import Deconv


@implementer(IOpenCLUnit)
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
        self.bias = None
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

    def initialize(self, device, **kwargs):
        if self.bias is not None:
            raise error.BadFormatError("bias should not be set")
        if self.err_output.mem is None:
            raise error.BadFormatError("err_output should be assigned")
        if self.weights.mem is None:
            raise error.BadFormatError("weights should be assigned")
        if self.input.mem is None:
            raise error.BadFormatError("input should be assigned")
        weights_shape = (list(
            self.weights.shape[i] for i in range(
                len(self.weights.shape) - 1, -1, -1))
            if self.weights_transposed else list(self.weights.shape))
        if (len(weights_shape) != 2 or
                weights_shape[0] != self.n_kernels or
                weights_shape[1] % (self.kx * self.ky) != 0):
            raise error.BadFormatError(
                "Incorrectly shaped weights encountered")
        if (len(self.input.shape) != 4 or
                self.input.shape[3] != self.n_kernels):
            raise error.BadFormatError(
                "Incorrectly shaped input encountered")
        if (len(self.err_output.shape) != 4 or
                self.err_output.shape[0] != self.input.shape[0]):
            raise error.BadFormatError(
                "Incorrectly shaped err_output encountered")

        batch_size = self.err_output.mem.shape[0]
        sy = self.err_output.mem.shape[1]
        sx = self.err_output.mem.shape[2]
        n_channels = self.err_output.size // (batch_size * sx * sy)
        n_weights = self.n_kernels * self.kx * self.ky * n_channels

        if self.weights.size != n_weights:
            raise error.BadFormatError(
                "Expected number of weights to match "
                "input, n_kernels, kx, ky parameters")

        try:
            padding = Deconv.compute_padding(
                sx, sy, self.kx, self.ky, self.sliding)
            if self.padding is None:
                self.padding = padding
            elif self.padding != padding:
                raise error.BadFormatError("Expected padding %s got %s" %
                                           (str(padding), str(self.padding)))
        except error.BadFormatError:
            if not self.hits:
                raise
            self.warning("Using unsafe padding of %s", str(self.padding))

        if self.hits:
            self.hits.initialize(self.device)

        super(GDDeconv, self).initialize(device, **kwargs)

    def ocl_init(self):
        batch_size = self.err_output.mem.shape[0]
        sy = self.err_output.mem.shape[1]
        sx = self.err_output.mem.shape[2]
        n_channels = self.err_output.size // (batch_size * sx * sy)
        kernel_applies_count = self.input.size // self.n_kernels
        kernel_size = self.kx * self.ky * n_channels
        dtype = self.err_output.mem.dtype

        self.cl_const = numpy.zeros(5, dtype=dtype)

        side = self.weights.shape[1 if self.weights_transposed else 0]
        other = self.weights.size // side
        assert side == self.n_kernels
        assert other == self.kx * self.ky * n_channels

        defines = {
            'INCLUDE_BIAS': 0,
            'H': other,
            'Y': side,
            'ACTIVATION_LINEAR': 1,
            'APPLY_GRADIENT': int(self.apply_gradient),
            'WEIGHTS_TRANSPOSED': int(self.weights_transposed),
            'ACCUMULATE_GRADIENT': int(self.accumulate_gradient),
            'BATCH': batch_size,
            'SX': sx,
            'SY': sy,
            'N_CHANNELS': n_channels,
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
            'USE_HITS': int(bool(self.hits))
        }

        a_width = kernel_applies_count
        b_width = self.n_kernels
        block_size = self.device.device_info.get_block_size(
            kernel="conv", dtype=self.err_output.dtype)
        self.cl_sources_["conv/forward"] = {
            "BLOCK_SIZE": block_size,
        }
        self._global_size_err_input = [
            roundup(b_width, block_size),
            roundup(a_width, block_size)]
        self._local_size_err_input = [block_size, block_size]

        a_width = kernel_size if self.weights_transposed else self.n_kernels
        b_width = self.n_kernels if self.weights_transposed else kernel_size
        block_size = self.device.device_info.get_block_size(
            kernel="conv", dtype=self.err_output.dtype)
        self.cl_sources_["deconv/gradient_descent/weights_update"] = {
            "BLOCK_SIZE": block_size,
            "USE_ORTHO": int(bool(self.factor_ortho)),
            'USE_MOMENT': int(bool(self.gradient_moment))
        }
        self._global_size_weights = [
            roundup(b_width, block_size),
            roundup(a_width, block_size)]
        self._local_size_weights = [block_size, block_size]

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

        if self.need_err_input:
            self.krn_err_input_ = self.get_kernel("feed_layer")
            self.krn_err_input_.set_args(self.err_output.devmem,
                                         self.weights.devmem,
                                         self.err_input.devmem)

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

    def gpu_err_output_update(self):
        self.err_output.unmap()
        self.execute_kernel([self.err_output.size], None, self.krn_err_output_)

    def gpu_err_input_update(self):
        if not self.need_err_input:
            return
        self.err_input.unmap()
        self.err_output.unmap()
        self.weights.unmap()

        self.execute_kernel(
            self._global_size_err_input, self._local_size_err_input,
            self.krn_err_input_)

    def ocl_run(self):
        self.gpu_err_output_update()
        self.gpu_err_input_update()
        self.gpu_weights_update()

    def cpu_run(self):
        raise RuntimeError("Not implemented")
