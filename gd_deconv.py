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
        padding = kwargs.get("padding", (0, 0, 0, 0))  # Left Top Right Bottom
        sliding = kwargs.get("sliding", (1, 1))  # X Y
        kwargs["n_kernels"] = n_kernels
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        kwargs["padding"] = padding
        kwargs["sliding"] = sliding
        super(GDDeconv, self).__init__(workflow, **kwargs)
        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.padding = tuple(padding)
        self.sliding = tuple(sliding)
        self.cl_const = None
        self.bias = None
        self.global_size_err_input = None
        self.local_size_err_input = None
        self.global_size_weights = None
        self.local_size_weights = None

    def init_unpickled(self):
        super(GDDeconv, self).init_unpickled()
        self.cl_sources_["gradient_descent_deconv.cl"] = {}
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

        if (self.err_input.mem is None or
                self.err_input.size != self.input.size):
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
        self.err_input.initialize(device)
        if self.store_gradient:
            self.gradient_weights.initialize(device)

        if device is None:
            return

        if self.program_ is None:
            self.cl_const = numpy.zeros(3, dtype=dtype)

            defines = {
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
                'SLIDE_Y': self.sliding[1]
            }
            my_defines = self.build_program(
                defines, "%s/gd_deconv_%d_%d.cl" % (
                    root.common.cache_dir,
                    self.input.mem.size // self.input.mem.shape[0],
                    self.err_output.mem.size // self.err_output.mem.shape[0]),
                dtype=dtype)

            self.krn_err_input_ = self.get_kernel("feed_layer")
            self.krn_err_input_.set_args(self.err_output.devmem,
                                         self.weights.devmem,
                                         self.err_input.devmem)

            self.krn_weights_ = self.get_kernel("weights_update")
            self.krn_weights_.set_args(self.err_output.devmem,
                                       self.input.devmem,
                                       self.weights.devmem,
                                       self.gradient_weights.devmem)

            block_size = self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(dtype)]
            if my_defines["BLOCK_SIZE"] != block_size:  # sanity check
                raise error.Bug("Returned BLOCK_SIZE differs from expected")

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

    def gpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return
        self.err_input.unmap()
        self.err_output.unmap()
        self.weights.unmap()

        self.execute_kernel(self.global_size_err_input,
                            self.local_size_err_input,
                            self.krn_err_input_)

    def cpu_err_input_update(self):
        # TODO(a.kazantsev): copy paste from Conv.cpu_run.
        if not self.need_err_input:
            return
        if self.weights_transposed:
            raise NotImplementedError(
                "cpu_run is not implemented for transposed weights")
        raise NotImplementedError()

    def gpu_weights_update(self):
        self.input.unmap()
        self.err_output.unmap()
        self.weights.unmap()
        self.gradient_weights.unmap()

        alpha_batch = -self.learning_rate
        alpha_lambda = -self.learning_rate * self.weights_decay

        self.cl_const[0] = alpha_batch
        self.cl_const[1] = alpha_lambda
        self.cl_const[2] = self.gradient_moment
        self.krn_weights_.set_args(cl.skip(4), self.cl_const[0:1],
                                   self.cl_const[1:2], self.cl_const[2:3])

        self.execute_kernel(self.global_size_weights, self.local_size_weights,
                            self.krn_weights_)

    def cpu_weights_update(self):
        # TODO(a.kazantsev): implement.
        if self.weights_transposed:
            raise NotImplementedError(
                "cpu_run is not implemented for transposed weights")
        raise NotImplementedError()

    def ocl_run(self):
        """Do gradient descent.
        """
        self.gpu_err_input_update()
        self.gpu_weights_update()

    def cpu_run(self):
        self.cpu_err_input_update()
        self.cpu_weights_update()
