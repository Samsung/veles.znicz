"""
Created on Jul 1, 2014

Deconvolutional layer.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import numpy
import scipy.signal
from zope.interface import implementer

from veles.config import root
from veles.opencl_units import IOpenCLUnit
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units
import veles.error as error


@implementer(IOpenCLUnit)
class Deconv(nn_units.Forward):
    """Deconvolutional layer for simple convolutional layer
    with linear activation and without bias.

    Must be assigned before initialize():
        input
        weights

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: input as batch of multichannel interleaved images.
        output: output as batch of multichannel interleaved images.
        weights: matrix of weights.
        n_kernels: number of convolutional kernels
                   in the corresponding convolutional layer.
        kx: kernel width.
        ky: kernel height.
        padding: tuple of virtual sample padding (left, top, right, bottom).
        sliding: tuple of kernel sliding (by x-axis, by y-axis).
        weights_transposed: assume weights matrix as a transposed one.
    """
    def __init__(self, workflow, **kwargs):
        try:
            n_kernels = kwargs["n_kernels"]
            kx = kwargs["kx"]
            ky = kwargs["ky"]
        except KeyError:
            raise KeyError("n_kernels, kx and ky are required parameters")
        padding = kwargs.get("padding", (0, 0, 0, 0))
        sliding = kwargs.get("sliding", (1, 1))
        kwargs["n_kernels"] = n_kernels
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        kwargs["padding"] = padding
        kwargs["sliding"] = sliding
        super(Deconv, self).__init__(workflow, **kwargs)
        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.padding = tuple(padding)
        self.sliding = tuple(sliding)
        self.bias = None
        self.exports.extend(("kx", "ky", "n_kernels", "padding", "sliding"))
        self.demand("input", "weights")
        self.global_size = None
        self.local_size = None

    def init_unpickled(self):
        super(Deconv, self).init_unpickled()
        # Reuse err_h_update kernel
        self.cl_sources_["gradient_descent_conv.cl"] = {}
        self.krn_clear_ = None

    def initialize(self, device, **kwargs):
        super(Deconv, self).initialize(device, **kwargs)

        if self.bias is not None:
            raise error.BadFormatError("bias should not be set")
        if self.input.mem is None:
            raise error.BadFormatError("input should be assigned")
        if (len(self.input.shape) != 4 or
                self.input.shape[3] != self.n_kernels):
            raise error.BadFormatError(
                "Incorrectly shaped input encountered")
        if self.weights.mem is None:
            raise error.BadFormatError("weights should be assigned")
        weights_shape = (list(
            self.weights.shape[i] for i in range(
                len(self.weights.shape) - 1, -1, -1))
            if self.weights_transposed else list(self.weights.shape))
        if (len(weights_shape) != 2 or
                weights_shape[0] != self.n_kernels or
                weights_shape[1] % (self.kx * self.ky) != 0):
            raise error.BadFormatError(
                "Incorrectly shaped weights encountered")

        dtype = self.input.mem.dtype

        batch_size = self.input.shape[0]
        output_channels = weights_shape[1] // (self.kx * self.ky)
        sy = self.input.shape[1]
        sx = self.input.shape[2]
        output_sy = ((sy - 1) * self.sliding[1] +
                     self.ky - self.padding[1] - self.padding[3])
        output_sx = ((sx - 1) * self.sliding[0] +
                     self.ky - self.padding[0] - self.padding[2])

        output_shape = [batch_size, output_sy, output_sx, output_channels]
        output_size = int(numpy.prod(output_shape))
        if self.output.mem is None or self.output.size != output_size:
            self.output.reset()
            if root.common.unit_test:
                output_shape[0] <<= 1
                self.output.mem = numpy.zeros(output_shape, dtype=dtype)
                self.output.initialize(device)
                self.output.vv = self.output.mem
                self.output.mem[batch_size:] = 1.0e30
                self.output.mem = self.output.mem[:batch_size]
                formats.assert_addr(self.output.mem, self.output.vv)
            else:
                self.output.mem = numpy.zeros(output_shape, dtype=dtype)
        else:
            self.output.mem.shape = output_shape
        del output_size
        del output_shape

        self.input.initialize(device)
        self.weights.initialize(device)
        self.output.initialize(device)

        if device is None:
            return

        defines = {
            'USE_ATOMICS': 1,
            'APPLY_GRADIENT': 0,
            'STORE_GRADIENT': 0,
            'INCLUDE_BIAS': 0,
            'WEIGHTS_TRANSPOSED': int(self.weights_transposed),
            'BATCH': batch_size,
            'SX': output_sx,
            'SY': output_sy,
            'N_CHANNELS': output_channels,
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
            defines, "%s/deconv_%dx%dx%d_%dx%d_%d.cl" % (
                root.common.cache_dir, output_sx, output_sy, output_channels,
                self.kx, self.ky, self.n_kernels), dtype=dtype)

        self.assign_kernel("err_h_update")
        self.set_args(self.input, self.weights, self.output)

        self.krn_clear_ = self.get_kernel("array_clear")
        self.krn_clear_.set_arg(0, self.output.devmem)

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(dtype)]
        if my_defines["BLOCK_SIZE"] != block_size:  # sanity check
            raise error.Bug("Returned BLOCK_SIZE differs from expected")
        self.global_size = [
            formats.roundup(weights_shape[1], block_size),
            formats.roundup(self.input.size // self.n_kernels, block_size)]
        self.local_size = [block_size, block_size]

    def ocl_run(self):
        # Clear the resulting matrix
        self.execute_kernel([self.output.mem.size], None, self.krn_clear_)
        self.execute_kernel(self.global_size, self.local_size)

    def cpu_run(self):
        if self.weights_transposed:
            raise NotImplementedError(
                "cpu_run is not implemented for transposed weights")

        self.output.map_invalidate()
        self.input.map_read()
        self.weights.map_read()

        batch_size = self.output.mem.shape[0]
        sy = self.output.mem.shape[1]
        sx = self.output.mem.shape[2]
        n_channels = self.output.mem.size // (batch_size * sx * sy)
        sx_full = self.padding[0] + sx + self.padding[2]
        sy_full = self.padding[1] + sy + self.padding[3]

        self.output.mem[:] = 0
        # initialize sparse output error
        sparse_input = numpy.zeros((
            batch_size, sy_full - self.ky + 1, sx_full - self.kx + 1,
            self.n_kernels), dtype=self.input.mem.dtype)
        for (batch, i, j, k), vle in numpy.ndenumerate(self.input.mem):
            sparse_input[batch, i * self.sliding[1],
                         j * self.sliding[0], k] = vle
        sample = numpy.empty((sy_full - self.ky + 1, sx_full - self.kx + 1))
        for batch, k in ((batch, k)
                         for batch in range(batch_size)
                         for k in range(self.n_kernels)):
            sample[:] = sparse_input[batch, :, :, k]
            cur_kernel = self.weights.mem[k].reshape(self.ky, self.kx,
                                                     n_channels)
            for ch in range(n_channels):
                output_full = scipy.signal.convolve2d(sample,
                                                      cur_kernel[:, :, ch],
                                                      mode='full')
                self.output.mem[batch, :, :, ch] += \
                    output_full[self.padding[1]:(sy_full - self.padding[3]),
                                self.padding[0]:(sx_full - self.padding[2])]
