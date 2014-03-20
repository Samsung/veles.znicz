"""
Created on Aug 27, 2013

Convolutional layers.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import logging
import numpy
import time

from veles.config import root
import veles.error as error
import veles.formats as formats
import veles.znicz.nn_units as nn_units


class Conv(nn_units.Forward):
    """Convolutional forward propagation with linear activation f(x) = x.

    Should be assigned before initialize():
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
        weights_magnitude: magnitude of the random distribution of weights.
        rand: rnd.Rand() object for initial weights generation.
        krn_: OpenCL kernel.
        s_activation: activation define for OpenCL source.
        weights_transposed: assume weights matrix as a transposed one.
    """
    def __init__(self, workflow, **kwargs):
        n_kernels = kwargs.get("n_kernels", 5)
        kx = kwargs.get("kx", 5)
        ky = kwargs.get("ky", 5)
        kwargs["n_kernels"] = n_kernels
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        super(Conv, self).__init__(workflow, **kwargs)
        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.s_activation = "ACTIVATION_LINEAR"
        self.exports.extend(("s_activation", "kx", "ky", "n_kernels"))

    def init_unpickled(self):
        super(Conv, self).init_unpickled()
        self.cl_sources_["conv.cl"] = {}
        self.krn_ = None

    def get_weights_magnitude(self):
        """
        Returns: weights magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        n_channels = (self.input.v.size // (self.input.v.shape[0] *
                      self.input.v.shape[1] * self.input.v.shape[2]))
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            return (1.0 / self.input.supposed_maxvle /
                (self.kx * self.ky * n_channels))
        return (9.0 / self.input.supposed_maxvle /
                (self.kx * self.ky * n_channels))

    def initialize(self):
        super(Conv, self).initialize()

        if self.weights_magnitude is None:
            # Get weights magnitude and cap it to 0.05
            self.weights_magnitude = min(self.get_weights_magnitude(), 0.05)
        batch_size = self.input.v.shape[0]
        sy = self.input.v.shape[1]
        sx = self.input.v.shape[2]
        n_channels = self.input.v.size // (batch_size * sx * sy)
        n_weights = self.n_kernels * self.kx * self.ky * n_channels
        if self.weights.v is None or self.weights.v.size != n_weights:
            self.weights.reset()
            self.weights.v = numpy.zeros(n_weights, dtype=self.input.v.dtype)
            self.rand.fill(self.weights.v, -self.weights_magnitude,
                           self.weights_magnitude)
            self.weights.v = self.weights.v.reshape(self.n_kernels,
                self.kx * self.ky * n_channels)
            # Reshape weights as a matrix:
            if self.weights_transposed:
                a = self.weights.v.transpose().copy()
                self.weights.v.shape = a.shape
                self.weights.v[:] = a[:]
        if (self.bias.v is None or
            self.bias.v.size != self.n_kernels):
            self.bias.reset()
            self.bias.v = numpy.zeros(self.n_kernels, dtype=self.input.v.dtype)
            self.rand.fill(self.bias.v, -self.weights_magnitude,
                           self.weights_magnitude)

        if root.common.unit_test:
            batch_size <<= 1  # check for overflow
        output_size = batch_size * (self.n_kernels *
            (sx - self.kx + 1) * (sy - self.ky + 1))
        if self.output.v is None or self.output.v.size != output_size:
            self.output.reset()
            self.output.v = numpy.zeros([batch_size,
                sy - self.ky + 1, sx - self.kx + 1, self.n_kernels],
                dtype=self.input.v.dtype)
        del output_size

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if root.common.unit_test:
            batch_size >>= 1
            self.output.vv = self.output.v
            self.output.v = self.output.v[:batch_size]
            formats.assert_addr(self.output.v, self.output.vv)

        if self.device is None:
            return

        if self.krn_ is None:
            defines = {
                self.s_activation: 1,
                'BLOCK_SIZE': self.device.device_info.BLOCK_SIZE[
                                                    root.common.precision_type],
                'BATCH': batch_size,
                'SX': sx,
                'SY': sy,
                'N_CHANNELS': n_channels,
                'KX': self.kx,
                'KY': self.ky,
                'N_KERNELS': self.n_kernels
            }
            if self.weights_transposed:
                defines['WEIGHTS_TRANSPOSED'] = 1
            self.build_program(defines, "%s/conv_%dx%dx%d_%dx%d_%d.cl" % (
                root.common.cache_dir, sx, sy, n_channels, self.kx, self.ky,
                self.n_kernels))

            self.krn_ = self.get_kernel("feed_layer")
            self.krn_.set_arg(0, self.input.v_)
            self.krn_.set_arg(1, self.weights.v_)
            self.krn_.set_arg(2, self.output.v_)
            self.krn_.set_arg(3, self.bias.v_)

    def print_times(self, t_start):
        """Show some statistics.
        """
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        self.output.map_read()
        y = self.output.v
        if y.dtype in (numpy.complex64, numpy.complex128):
            self.debug("%s: %d samples with %d weights in %.2f sec: "
                "y: min avg max: %.6f %.6f %.6f" %
                (self.__class__.__name__, y.shape[0],
                 self.weights.v.size, time.time() - t_start,
                 min(y.real.min(), y.imag.min()),
                 (numpy.average(y.real) + numpy.average(y.imag)) * 0.5,
                 max(y.real.max(), y.imag.max())))
        else:
            self.debug("%s: %d samples with %d weights in %.2f sec: "
                "y: min avg max: %.6f %.6f %.6f" %
                (self.__class__.__name__, y.shape[0],
                 self.weights.v.size, time.time() - t_start,
                 y.min(), numpy.average(y), y.max()))

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()  # we will be updating output
        self.input.unmap()  # we will use input
        self.weights.unmap()  # we will use weights
        self.bias.unmap()  # we will use bias
        block_size = self.device.device_info.BLOCK_SIZE[root.common.precision_type]
        global_size = [formats.roundup(self.n_kernels, block_size),
                       formats.roundup(self.output.v.size // self.n_kernels,
                                       block_size)]
        local_size = [block_size, block_size]
        event = self.execute_kernel(self.krn_, global_size, local_size)
        event.wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        raise error.ErrNotImplemented()

    def run(self):
        t1 = time.time()
        retval = super(Conv, self).run()
        if retval:
            return retval
        self.print_times(t1)


class ConvTanh(Conv):
    """Conv with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).
    """
    def initialize(self):
        self.s_activation = "ACTIVATION_TANH"
        super(ConvTanh, self).initialize()
        self.output.supposed_maxvle = 1.7159

    def get_weights_magnitude(self):
        """
        Returns: weights magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        n_channels = (self.input.v.size // (self.input.v.shape[0] *
                      self.input.v.shape[1] * self.input.v.shape[2]))
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            return (1.0 / (self.input.supposed_maxvle * 0.6666) /
                (self.kx * self.ky * n_channels))
        return (9.0 / (self.input.supposed_maxvle * 0.6666) /
                (self.kx * self.ky * n_channels))


class ConvRELU(Conv):
    """Conv with RELU activation f(x) = log(1.0 + exp(x)).
    """
    def initialize(self):
        self.s_activation = "ACTIVATION_TANH"
        super(ConvTanh, self).initialize()
        self.output.supposed_maxvle = 10
