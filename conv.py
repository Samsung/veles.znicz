"""
Created on Aug 27, 2013

Convolutional layers.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import logging
import math
from math import pi
import numpy
import time
from zope.interface import implementer

from veles.config import root
from veles.opencl_units import IOpenCLUnit

import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
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
        weights_filling: what weight filling to use: `"uniform"` for uniform
            random distribution, `"normal"` for normal distribution,
            `"gabor"` for using Gabor kernels.
        weights_stddev: standard deviation of normal or Gabor weight fillings

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
        n_channels = (self.input.mem.size // (self.input.mem.shape[0] *
                      self.input.mem.shape[1] * self.input.mem.shape[2]))
        vle = (1.0 / self.input.supposed_maxvle /
               numpy.sqrt(self.kx * self.ky * n_channels))
        if self.weights_filling == "gaussian":
            vle /= 3
        return vle

    def initialize(self, device, **kwargs):
        super(Conv, self).initialize(device=device, **kwargs)

        if self.weights_stddev is None:
            self.weights_stddev = min(self.get_weights_magnitude(), 0.05)
        if self.bias_stddev is None:
            self.bias_stddev = self.weights_stddev

        self._batch_size = self.input.mem.shape[0]
        self._sy = self.input.mem.shape[1]
        self._sx = self.input.mem.shape[2]
        self._n_channels = (self.input.mem.size //
                            (self._batch_size * self._sx * self._sy))

        self._fill_weights()
        self._fill_biases()

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
        if self.output.mem is None or self.output.mem.size != output_size:
            self.output.reset()
            self.output.mem = numpy.zeros(output_shape,
                                          dtype=self.input.mem.dtype)
        del output_size
        del output_shape

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if root.common.unit_test:
            self._batch_size >>= 1
            self.output.vv = self.output.mem
            self.output.mem = self.output.mem[:self._batch_size]
            formats.assert_addr(self.output.mem, self.output.vv)

        if self.device is None:
            return

        defines = {
            self.s_activation: 1,
            'WEIGHTS_TRANSPOSED': int(self.weights_transposed),
            'INCLUDE_BIAS': int(self.include_bias),
            'BLOCK_SIZE': self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.input.mem.dtype)],
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
        self.build_program(defines, "%s/conv_%dx%dx%d_%dx%d_%d.cl" % (
            root.common.cache_dir, self._sx, self._sy, self._n_channels,
            self.kx, self.ky, self.n_kernels),
            dtype=self.input.mem.dtype)

        self.assign_kernel("feed_layer")
        if self.include_bias:
            self.set_args(self.input, self.weights, self.bias, self.output)
        else:
            self.set_args(self.input, self.weights, self.output)

    def print_debug_data(self, t_start):
        """Show some statistics.
        """
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        self.output.map_read()
        y = self.output.mem
        if y.dtype in (numpy.complex64, numpy.complex128):
            self.debug(
                "%s: %d samples with %d weights in %.2f sec: "
                "y: min avg max: %.6f %.6f %.6f" %
                (self.__class__.__name__, y.shape[0],
                 self.weights.mem.size, time.time() - t_start,
                 min(y.real.min(), y.imag.min()),
                 (numpy.average(y.real) + numpy.average(y.imag)) * 0.5,
                 max(y.real.max(), y.imag.max())))
        else:
            self.debug(
                "%s: %d samples with %d weights in %.2f sec: "
                "y: min avg max: %.6f %.6f %.6f" %
                (self.__class__.__name__, y.shape[0],
                 self.weights.mem.size, time.time() - t_start,
                 y.min(), numpy.average(y), y.max()))

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()  # we will be updating output
        self.input.unmap()  # we will use input
        self.weights.unmap()  # we will use weights
        self.bias.unmap()  # we will use bias
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.mem.dtype)]
        global_size = [formats.roundup(self.n_kernels, block_size),
                       formats.roundup(self.output.mem.size // self.n_kernels,
                                       block_size)]
        local_size = [block_size, block_size]
        self.execute_kernel(global_size, local_size)

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

        assert self.kx >= 0 and self.ky >= 0
        for batch, _ in ((batch, ch)
                         for batch in range(self._batch_size)
                         for ch in range(self._n_channels)):
            for k, kernel in enumerate(self.weights.mem):
                for i, j in ((i, j) for i in range(ny) for j in range(nx)):
                    full_i1 = i * self.sliding[1]
                    full_i2 = full_i1 + self.ky
                    full_j1 = j * self.sliding[0]
                    full_j2 = full_j1 + self.kx
                    in_i1 = min(max(full_i1 - self.padding[1], 0), self._sy)
                    in_i2 = min(max(full_i2 - self.padding[1], 0), self._sy)
                    in_j1 = min(max(full_j1 - self.padding[0], 0), self._sx)
                    in_j2 = min(max(full_j2 - self.padding[0], 0), self._sx)
                    cut_i1, cut_i2 = (in_i1 - full_i1 + self.padding[1],
                                      in_i2 - full_i1 + self.padding[1])
                    cut_j1, cut_j2 = (in_j1 - full_j1 + self.padding[0],
                                      in_j2 - full_j1 + self.padding[0])
                    if in_i2 - in_i1 > 0 or in_j2 - in_j1 > 0:
                        cut = self.input.mem[batch, in_i1:in_i2, in_j1:in_j2]
                        kernel_3d = kernel.reshape(self.ky, self.kx,
                                                   self._n_channels)
                        cutted_kernel = kernel_3d[cut_i1:cut_i2,
                                                  cut_j1:cut_j2, :]
                        assert cut.size == cutted_kernel.size
                        conv = numpy.sum(numpy.multiply(cut.ravel(),
                                                        cutted_kernel.ravel()))
                        self.output.mem[batch, i, j, k] = conv
        # add bias and apply activation function
        self.apply_activation()

    def run(self):
        t1 = time.time()
        retval = super(Conv, self).run()
        if retval:
            return retval
        self.print_debug_data(t1)

    def apply_activation(self):
        """Add bias and apply linear activation function.
        """
        assert self.s_activation == "ACTIVATION_LINEAR"
        if self.include_bias:
            self.output.mem += self.bias.mem

    def _fill_weights(self):
        """
        Fills initial filter weights according to `weights_filling` attribute.
        Called within ``initialize`` method.
        """
        n_weights = self.n_kernels * self.kx * self.ky * self._n_channels
        if self.weights.mem is None or self.weights.mem.size != n_weights:
            self.weights.reset()
            self.weights.mem = numpy.zeros(n_weights,
                                           dtype=self.input.mem.dtype)
            if self.weights_filling == "uniform":
                self.rand.fill(self.weights.mem, -self.weights_stddev,
                               self.weights_stddev)
            elif self.weights_filling == "gaussian":
                self.rand.fill_normal_real(self.weights.mem, 0,
                                           self.weights_stddev)
            elif self.weights_filling == "constant":
                self.weights.mem[:] = self.weights_stddev
            elif self.weights_filling == "gabor":
                self._fill_with_gabor_filters(
                    self.n_kernels, (self.ky, self.kx), self.weights_stddev)
            else:
                raise error.BadFormatError("Invalid weights filling type")
            self.weights.mem = self.weights.mem.reshape(
                self.n_kernels, self.kx * self.ky * self._n_channels)
            # Reshape weights as a matrix:
            if self.weights_transposed:
                a = self.weights.mem.transpose().copy()
                self.weights.mem.shape = a.shape
                self.weights.mem[:] = a[:]

    def _fill_biases(self):
        """
        Fills filter biases according to `bias_filling` attribute.
        Called within ``initialize`` method.
        """
        if not self.include_bias:
            return
        if (self.bias.mem is None or
                self.bias.mem.size != self.n_kernels):
            self.bias.reset()
            self.bias.mem = numpy.zeros(self.n_kernels,
                                        dtype=self.input.mem.dtype)
            if self.bias_filling == "uniform":
                self.rand.fill(self.bias.mem, -self.bias_stddev,
                               self.bias_stddev)
            elif self.bias_filling == "gaussian":
                self.rand.fill_normal_real(self.bias.mem, 0, self.bias_stddev)
            elif self.bias_filling == "constant":
                self.bias.mem[:] = self.bias_stddev
            else:
                raise error.BadFormatError("Invalid bias filling type")

    def _fill_with_gabor_filters(self, n_filters, shape, stddev):
        """
        Fills weights and biases with Gabor filters. Only 96 filters
            are implemented now, others are filled with white noise.

        Args:
            n_filters(int): number of filters
            shape(tuple): shape of each filter
            stddev(float): standard deviation of filtering kernels
        """
        import cv2  # TODO(a.kazantsev): implement getGaborKernel manually
                    # and remove this dependency.

        #Gabor  filters
        orientations = [0, pi / 4, pi / 2, 3 * pi / 4]  # tilt of filters
        phase_shifts = [0, pi]  # pi phase shift inverts signal

        size = min(shape)

        kernels_count = 0
        n_chans = self.weights.mem.size // (self.kx * self.ky * self.n_kernels)
        for wavelen_ratio in range(4):  # how much waves should lay in kernel
            for dev_ratio in range(1, 2 * wavelen_ratio + 1):
                for ori in orientations:
                    for phase in phase_shifts:
                        kernel_chan = cv2.getGaborKernel(
                            ksize=shape, sigma=size / dev_ratio / 2,
                            theta=ori, lambd=size / wavelen_ratio,
                            gamma=1, psi=phase)

                        kernel_chan = formats.norm_image(kernel_chan) * stddev
                        kernel = numpy.zeros(shape=[n_chans, self.kx, self.ky],
                                             dtype=numpy.float64)
                        for chan in range(n_chans):
                            kernel[chan, :] = kernel_chan
                        kernel = kernel.swapaxes(0, 2)
                        self.weights.mem[
                            kernels_count * kernel.size:
                            (kernels_count + 1) * kernel.size] = kernel.ravel()

                        kernels_count += 1
                        if kernels_count == n_filters:
                            return

        #White noise (if more, than 96 filters are required)
        self.rand.fill_normal_real(self.weights.mem[kernels_count:], 0, stddev)


class ConvTanh(Conv):
    """Conv with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).
    """
    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_TANH"
        super(ConvTanh, self).initialize(device=device, **kwargs)
        self.output.supposed_maxvle = 1.7159

    def apply_activation(self):
        """Add bias and apply tanh activation function.
        """
        assert self.s_activation == "ACTIVATION_TANH"
        for k in range(self.n_kernels):
            for x in numpy.nditer(self.output.mem[:, :, :, k],
                                  op_flags=['readwrite']):
                x[...] = math.tanh((x + self.bias.mem[k]) * 0.6666) * 1.7159


class ConvRELU(Conv):
    """Conv with smooth RELU activation f(x) = log(1.0 + exp(x)).
    """
    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_RELU"
        super(ConvRELU, self).initialize(device=device, **kwargs)
        self.output.supposed_maxvle = 10

    def apply_activation(self):
        """Add bias and apply RELU activation function.
        """
        assert self.s_activation == "ACTIVATION_RELU"
        for k in range(self.n_kernels):
            for x in numpy.nditer(self.output.mem[:, :, :, k],
                                  op_flags=['readwrite']):
                tmp_val = x + self.bias.mem[k]
                if tmp_val > 15:
                    x[...] = tmp_val
                else:
                    x[...] = math.log(math.exp(tmp_val) + 1)


class ConvStrictRELU(Conv):
    """
    Conv with strict RELU activation f(x) = (x >= 0) ? x : 0
    (Just like in CAFFE)
    """
    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_STRICT_RELU"
        super(ConvStrictRELU, self).initialize(device=device, **kwargs)
        self.output.supposed_maxvle = 10

    def apply_activation(self):
        """Add bias and apply STRICT_RELU activation function.
        """
        assert self.s_activation == "ACTIVATION_STRICT_RELU"
        for k in range(self.n_kernels):
            for x in numpy.nditer(self.output.mem[:, :, :, k],
                                  op_flags=['readwrite']):
                tmp_val = x + self.bias.mem[k]
                x[...] = numpy.where(numpy.greater(tmp_val, 0), tmp_val, 0)
