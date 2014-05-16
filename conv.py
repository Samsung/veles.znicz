"""
Created on Aug 27, 2013

Convolutional layers.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import math
import numpy
import time

from veles.config import root

import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units


class Conv(nn_units.Forward):
    """Convolutional forward propagation with linear activation f(x) = x.

    Must be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        weights
        bias
        output

    Attributes:
        input: input as batch of multichannel interleaved images.
        output: output as batch of multichannel interleaved images.
        weights: matrix of weights.
        bias: bias.
        n_kernels: number of convolutional kernels.
        kx: kernel width.
        ky: kernel height.
        padding: tuple of virtual sample padding (left, top, right, bottom).
        sliding: tuple of kernel sliding (by x-axis, by y-axis).
        weights_filling: rand weight filling
                         ("unirofm" (default) or "gaussian")
        weights_stddev: magnitude of uniform weight distribution.
        weights_stddev: StdDev of normal weight distributtion

        rand: rnd.Rand() object for initial weights generation.
        s_activation: activation define for OpenCL source.
        weights_transposed: assume weights matrix as a transposed one.
    """
    def __init__(self, workflow, **kwargs):
        n_kernels = kwargs.get("n_kernels", 5)
        kx = kwargs.get("kx", 5)
        ky = kwargs.get("ky", 5)
        padding = kwargs.get("padding", (0, 0, 0, 0))
        sliding = kwargs.get("sliding", (1, 1))
        kwargs["n_kernels"] = n_kernels
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        kwargs["padding"] = padding
        kwargs["sliding"] = sliding
        super(Conv, self).__init__(workflow, **kwargs)
        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.padding = tuple(padding)
        self.sliding = tuple(sliding)
        self.s_activation = "ACTIVATION_LINEAR"
        self.exports.extend(("s_activation", "kx", "ky", "n_kernels",
                             "padding", "sliding"))

    def init_unpickled(self):
        super(Conv, self).init_unpickled()
        self.cl_sources_["conv.cl"] = {}

    def get_weights_magnitude(self):
        """
        Returns: weights magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        n_channels = (self.input.v.size // (self.input.v.shape[0] *
                      self.input.v.shape[1] * self.input.v.shape[2]))
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            vle = (1.0 / self.input.supposed_maxvle /
                   (self.kx * self.ky * n_channels))
        else:
            vle = (9.0 / self.input.supposed_maxvle /
                   (self.kx * self.ky * n_channels))
        if self.weights_filling == "gaussian":
            vle /= 3
        return vle

    def initialize(self, device, **kwargs):
        super(Conv, self).initialize(device=device, **kwargs)

        if self.weights_stddev is None:
            self.weights_stddev = min(self.get_weights_magnitude(), 0.05)
        if self.bias_stddev is None:
            self.bias_stddev = self.weights_stddev

        self._batch_size = self.input.v.shape[0]
        self._sy = self.input.v.shape[1]
        self._sx = self.input.v.shape[2]
        self._n_channels = self.input.v.size // (self._batch_size * self._sx *
                                                 self._sy)
        n_weights = self.n_kernels * self.kx * self.ky * self._n_channels
        if self.weights.v is None or self.weights.v.size != n_weights:
            self.weights.reset()
            self.weights.v = numpy.zeros(n_weights, dtype=self.input.v.dtype)
            if self.weights_filling == "uniform":
                self.rand.fill(self.weights.v, -self.weights_stddev,
                               self.weights_stddev)
            elif self.weights_filling == "gaussian":
                self.rand.fill_normal_real(self.weights.v, 0,
                                           self.weights_stddev)
            elif self.weights_filling == "constant":
                self.weights.v[:] = self.weights_stddev
            else:
                raise error.ErrBadFormat("Invalid weights filling type")
            self.weights.v = self.weights.v.reshape(
                self.n_kernels, self.kx * self.ky * self._n_channels)
            # Reshape weights as a matrix:
            if self.weights_transposed:
                a = self.weights.v.transpose().copy()
                self.weights.v.shape = a.shape
                self.weights.v[:] = a[:]
        if (self.bias.v is None or
                self.bias.v.size != self.n_kernels):
            self.bias.reset()
            self.bias.v = numpy.zeros(self.n_kernels, dtype=self.input.v.dtype)
            if self.bias_filling == "uniform":
                self.rand.fill(self.bias.v, -self.bias_stddev,
                               self.bias_stddev)
            elif self.bias_filling == "gaussian":
                self.rand.fill_normal_real(self.bias.v, 0, self.bias_stddev)
            elif self.bias_filling == "constant":
                self.bias.v[:] = self.bias_stddev
            else:
                raise error.ErrBadFormat("Invalid bias filling type")

        if root.common.unit_test:
            self._batch_size <<= 1  # check for overflow
        output_shape = [
            self._batch_size,
            (self._sy + self.padding[1] + self.padding[3] - self.ky) //
            self.sliding[1] + 1,
            (self._sx + self.padding[0] + self.padding[2] - self.kx) //
            self.sliding[0] + 1,
            self.n_kernels]
        output_size = int(numpy.prod(output_shape))
        if self.output.v is None or self.output.v.size != output_size:
            self.output.reset()
            self.output.v = numpy.zeros(output_shape, dtype=self.input.v.dtype)
        del output_size
        del output_shape

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if root.common.unit_test:
            self._batch_size >>= 1
            self.output.vv = self.output.v
            self.output.v = self.output.v[:self._batch_size]
            formats.assert_addr(self.output.v, self.output.vv)

        if self.device is None:
            return

        defines = {
            self.s_activation: 1,
            'BLOCK_SIZE': self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)],
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
            'SLIDE_Y': self.sliding[1]
        }
        if self.weights_transposed:
            defines['WEIGHTS_TRANSPOSED'] = 1
        self.build_program(defines, "%s/conv_%dx%dx%d_%dx%d_%d.cl" % (
            root.common.cache_dir, self._sx, self._sy, self._n_channels,
            self.kx, self.ky, self.n_kernels),
            dtype=self.input.v.dtype)

        self.assign_kernel("feed_layer")
        self.set_args(self.input, self.weights, self.output, self.bias)

    def print_debug_data(self, t_start):
        """Show some statistics.
        """
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        self.output.map_read()
        y = self.output.v
        if y.dtype in (numpy.complex64, numpy.complex128):
            self.debug(
                "%s: %d samples with %d weights in %.2f sec: "
                "y: min avg max: %.6f %.6f %.6f" %
                (self.__class__.__name__, y.shape[0],
                 self.weights.v.size, time.time() - t_start,
                 min(y.real.min(), y.imag.min()),
                 (numpy.average(y.real) + numpy.average(y.imag)) * 0.5,
                 max(y.real.max(), y.imag.max())))
        else:
            self.debug(
                "%s: %d samples with %d weights in %.2f sec: "
                "y: min avg max: %.6f %.6f %.6f" %
                (self.__class__.__name__, y.shape[0],
                 self.weights.v.size, time.time() - t_start,
                 y.min(), numpy.average(y), y.max()))

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()  # we will be updating output
        self.input.unmap()  # we will use input
        self.weights.unmap()  # we will use weights
        self.bias.unmap()  # we will use bias
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)]
        global_size = [formats.roundup(self.n_kernels, block_size),
                       formats.roundup(self.output.v.size // self.n_kernels,
                                       block_size)]
        local_size = [block_size, block_size]
        self.execute_kernel(global_size, local_size).wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.input.map_read()
        self.weights.map_read()
        self.bias.map_read()
        self.output.map_invalidate()

        sx_full = self.padding[0] + self._sx + self.padding[2]
        sy_full = self.padding[1] + self._sy + self.padding[3]
        nx = (sx_full - self.kx) // self.sliding[0] + 1
        ny = (sy_full - self.ky) // self.sliding[1] + 1

        assert(self.kx >= 0 and self.ky >= 0)
        for batch, _ in ((batch, ch)
                         for batch in range(self._batch_size)
                         for ch in range(self._n_channels)):
            for k, kernel in enumerate(self.weights.v):
                for i, j in ((i, j) for i in range(ny) for j in range(nx)):
                    y1, y2 = (i * self.sliding[1],
                              i * self.sliding[1] + self.ky)
                    x1, x2 = (j * self.sliding[0],
                              j * self.sliding[0] + self.kx)
                    i1, i2 = (min(max(y1 - self.padding[1], 0), self._sy),
                              min(max(y2 - self.padding[1], 0), self._sy))
                    j1, j2 = (min(max(x1 - self.padding[0], 0), self._sx),
                              min(max(x2 - self.padding[0], 0), self._sx))
                    if i2 - i1 > 0 or j2 - j1 > 0:
                        cut = self.input.v[batch, i1:i2, j1:j2]
                        kernel_2d = kernel.reshape(self.ky, self.kx)
                        cutted_kernel = kernel_2d[(i1 - y1):(i2 - y1),
                                                  (j1 - x1):(j2 - x1)]
                        assert(cut.size == cutted_kernel.size)
                        conv = numpy.sum(numpy.multiply(cut.ravel(),
                                                        cutted_kernel.ravel()))
                        self.output.v[batch, i, j, k] = conv

                        # add bias and apply activation function
                        if self.s_activation == "ACTIVATION_LINEAR":
                            self.output.v[batch, i, j, k] = (conv +
                                                             self.bias.v[k])
                        elif self.s_activation == "ACTIVATION_TANH":
                            self.output.v[batch, i, j, k] = \
                                math.tanh((conv + self.bias.v[k])
                                          * 0.6666) * 1.7159
                        elif self.s_activation == "ACTIVATION_RELU":
                            tmp_val = conv + self.bias.v[k]
                            if tmp_val > 15:
                                self.output.v[batch, i, j, k] = tmp_val
                            else:
                                self.output.v[batch, i, j, k] = \
                                    math.log(math.exp(tmp_val) + 1)
                        else:
                            raise ValueError("unknown type of activation "
                                             "function")
                    else:
                        break

    def run(self):
        t1 = time.time()
        retval = super(Conv, self).run()
        if retval:
            return retval
        self.print_debug_data(t1)


class ConvTanh(Conv):
    """Conv with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).
    """
    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_TANH"
        super(ConvTanh, self).initialize(device=device, **kwargs)
        self.output.supposed_maxvle = 1.7159

    def get_weights_magnitude(self):
        """
        Returns: weights magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        self._n_channels = (self.input.v.size //
                            numpy.prod(self.input.v.shape[:3]))
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            vle = (1.0 / (self.input.supposed_maxvle * 0.6666) /
                   (self.kx * self.ky * self._n_channels))
        else:
            vle = (9.0 / (self.input.supposed_maxvle * 0.6666) /
                   (self.kx * self.ky * self._n_channels))
        if self.weights_filling == "gaussian":
            vle /= 3
        return vle


class ConvRELU(Conv):
    """Conv with RELU activation f(x) = log(1.0 + exp(x)).
    """
    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_RELU"
        super(ConvRELU, self).initialize(device=device, **kwargs)
        self.output.supposed_maxvle = 10
