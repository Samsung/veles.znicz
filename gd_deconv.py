"""
Created on Jul 2, 2014

Gradient descent for deconvolutional layer.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import numpy
import opencl4py as cl
from zope.interface import implementer

from veles.config import root
import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.opencl_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units
from veles.znicz.deconv import Deconv


@implementer(IOpenCLUnit)
class GDDeconv(nn_units.GradientDescentBase):
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
    def __init__(self, workflow, **kwargs):
        try:
            n_kernels = kwargs["n_kernels"]
            kx = kwargs["kx"]
            ky = kwargs["ky"]
        except KeyError:
            raise KeyError("n_kernels, kx and ky are required parameters")
        super(GDDeconv, self).__init__(workflow, **kwargs)
        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.padding = kwargs.get("padding")
        self.sliding = tuple(kwargs.get("sliding", (1, 1)))
        self.cl_const = None
        self.bias = None
        self.global_size_err_input = None
        self.local_size_err_input = None
        self.global_size_weights = None
        self.local_size_weights = None
        self.reduce_size = 64

    def init_unpickled(self):
        super(GDDeconv, self).init_unpickled()
        self.cl_sources_["gradient_descent_deconv.cl"] = {}
        self.krn_err_output_ = None
        self.krn_err_input_ = None
        self.krn_weights_ = None

    def initialize(self, device, **kwargs):
        super(GDDeconv, self).initialize(device, **kwargs)

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

        dtype = self.err_output.mem.dtype

        batch_size = self.err_output.mem.shape[0]
        sy = self.err_output.mem.shape[1]
        sx = self.err_output.mem.shape[2]
        n_channels = self.err_output.size // (batch_size * sx * sy)
        n_weights = self.n_kernels * self.kx * self.ky * n_channels

        if self.weights.size != n_weights:
            raise error.BadFormatError(
                "Expected number of weights to match "
                "input, n_kernels, kx, ky parameters")

        padding = Deconv.compute_padding(
            sx, sy, self.kx, self.ky, self.sliding)
        if self.padding is None:
            self.padding = padding
        elif self.padding != padding:
            raise error.BadFormatError("Expected padding %s got %s" %
                                       (str(padding), str(self.padding)))

        if (self.need_err_input and (
                self.err_input.mem is None or
                self.err_input.size != self.input.size)):
            self.err_input.reset()
            sh = self.input.shape
            if root.common.unit_test:
                sh = list(sh)
                sh[0] <<= 1
                self.err_input.mem = numpy.zeros(sh, dtype=dtype)
                self.err_input.initialize(device)
                self.err_input.map_write()
                self.err_input.vv = self.err_input.mem
                self.err_input.mem[batch_size:] = numpy.nan
                self.err_input.mem = self.err_input.mem[:batch_size]
                formats.assert_addr(self.err_input.mem, self.err_input.vv)
            else:
                self.err_input.mem = numpy.zeros(sh, dtype=dtype)

        if (self.store_gradient and
                (self.gradient_weights.mem is None or
                 self.gradient_weights.size != self.weights.size)):
            self.gradient_weights.reset()
            self.gradient_weights.mem = numpy.zeros_like(self.weights.mem)

        self.weights.initialize(device)
        self.input.initialize(device)
        self.err_output.initialize(device)
        if self.need_err_input:
            self.err_input.initialize(device)
        if self.store_gradient:
            self.gradient_weights.initialize(device)

        if device is None:
            return

        self.cl_const = numpy.zeros(5, dtype=dtype)

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(dtype)]

        side = self.weights.shape[1 if self.weights_transposed else 0]
        other = self.weights.size // side

        if self.factor_ortho:
            self.reduce_size = min(self.reduce_size,
                                   max(self.weights.shape[0],
                                       self.weights.shape[1]))
            if not self.wwt or self.wwt.size < side * side:
                self.wwt.reset()
                self.wwt.mem = numpy.zeros([side, side], dtype=dtype)
            if not self.row_sums or self.row_sums.size < side:
                self.row_sums.reset()
                self.row_sums.mem = numpy.zeros(side, dtype=dtype)
            if not self.col_sums or self.col_sums.size < other:
                self.col_sums.reset()
                self.col_sums.mem = numpy.zeros(other, dtype=dtype)

            self.wwt.initialize(self.device)
            self.row_sums.initialize(self.device)
            self.col_sums.initialize(self.device)
        else:
            self.reduce_size = min(self.reduce_size, other)
        self.reduce_size = formats.roundup(self.reduce_size, 32)

        defines = {
            'USE_ORTHO': int(bool(self.factor_ortho)),
            'H': other,
            'Y': side,
            'ACTIVATION_LINEAR': 1,
            'APPLY_GRADIENT': int(self.apply_gradient),
            'WEIGHTS_TRANSPOSED': int(self.weights_transposed),
            'STORE_GRADIENT': int(self.store_gradient),
            'INCLUDE_BIAS': 0,
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
            'REDUCE_SIZE': self.reduce_size
        }
        my_defines = self.build_program(
            defines, "%s/gd_deconv_%d_%d.cl" % (
                root.common.cache_dir,
                self.input.mem.size // self.input.mem.shape[0],
                self.err_output.mem.size // self.err_output.mem.shape[0]),
            dtype=dtype)

        self.krn_err_output_ = self.get_kernel("err_output_update")
        self.krn_err_output_.set_arg(0, self.err_output.devmem)

        if self.need_err_input:
            self.krn_err_input_ = self.get_kernel("feed_layer")
            self.krn_err_input_.set_args(self.err_output.devmem,
                                         self.weights.devmem,
                                         self.err_input.devmem)

        self.krn_weights_ = self.get_kernel("weights_update")
        self.krn_weights_.set_args(self.err_output.devmem,
                                   self.input.devmem,
                                   self.weights.devmem,
                                   self.gradient_weights.devmem)

        if my_defines["BLOCK_SIZE"] != block_size:  # sanity check
            raise error.Bug("Returned BLOCK_SIZE differs from expected")

        if self.need_err_input:
            self.global_size_err_input = [
                formats.roundup(self.n_kernels, block_size),
                formats.roundup(self.err_input.mem.size // self.n_kernels,
                                block_size)]
            self.local_size_err_input = [block_size, block_size]

        if self.weights_transposed:
            self.global_size_weights = [
                formats.roundup(self.n_kernels, block_size),
                formats.roundup(self.kx * self.ky * n_channels,
                                block_size)]
        else:
            self.global_size_weights = [
                formats.roundup(self.kx * self.ky * n_channels,
                                block_size),
                formats.roundup(self.n_kernels, block_size)]
        self.local_size_weights = [block_size, block_size]

        if self.factor_ortho:
            self.krn_compute_wwt_ = self.get_kernel("compute_wwt")
            self.krn_compute_wwt_.set_args(self.weights.devmem,
                                           self.wwt.devmem)
            self.krn_compute_row_sums_ = self.get_kernel("compute_row_sums")
            self.krn_compute_row_sums_.set_args(self.wwt.devmem,
                                                self.row_sums.devmem)
            self.krn_compute_col_sums_ = self.get_kernel("compute_col_sums")
            self.krn_compute_col_sums_.set_args(self.weights.devmem,
                                                self.col_sums.devmem)

            self.krn_weights_.set_arg(9, self.row_sums.devmem)
            self.krn_weights_.set_arg(10, self.col_sums.devmem)

            self.info("Using orthogonalization factor of %.6f",
                      self.factor_ortho)

    def gpu_err_output_update(self):
        self.err_output.unmap()
        self.execute_kernel([self.err_output.size], None, self.krn_err_output_)

    def gpu_err_input_update(self):
        if not self.need_err_input:
            return
        self.err_input.unmap()
        self.err_output.unmap()
        self.weights.unmap()

        self.execute_kernel(self.global_size_err_input,
                            self.local_size_err_input,
                            self.krn_err_input_)

    def gpu_weights_update(self):
        self.input.unmap()
        self.err_output.unmap()
        self.weights.unmap()
        self.gradient_weights.unmap()

        lr = self.learning_rate
        factor_l12 = self.weights_decay
        l1_vs_l2 = self.l1_vs_l2

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_output.mem.dtype)]

        local_size = [block_size, block_size]
        if self.factor_ortho:
            self.wwt.unmap()
            self.row_sums.unmap()
            self.col_sums.unmap()
            side = self.wwt.shape[0]
            other = self.weights.size // side
            self.execute_kernel(
                [formats.roundup(side, block_size),
                 formats.roundup(side, block_size)],
                local_size, self.krn_compute_wwt_)
            self.execute_kernel(
                [side * self.reduce_size], [self.reduce_size],
                self.krn_compute_row_sums_)
            self.execute_kernel(
                [other * self.reduce_size], [self.reduce_size],
                self.krn_compute_col_sums_)

            self.cl_const[4] = self.factor_ortho
            self.krn_weights_.set_arg(8, self.cl_const[4:5])

        self.cl_const[0] = lr
        self.cl_const[1] = factor_l12
        self.cl_const[2] = l1_vs_l2
        self.cl_const[3] = self.gradient_moment
        self.krn_weights_.set_args(
            cl.skip(4), self.cl_const[0:1], self.cl_const[1:2],
            self.cl_const[2:3], self.cl_const[3:4])

        self.execute_kernel(self.global_size_weights, self.local_size_weights,
                            self.krn_weights_)

    def ocl_run(self):
        self.gpu_err_output_update()
        self.gpu_err_input_update()
        self.gpu_weights_update()

    def cpu_run(self):
        raise RuntimeError("Not implemented")
