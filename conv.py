"""
Created on Aug 27, 2013

Convolutional layers.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import nn_units
import formats
import numpy
import pyopencl
import time
import rnd
import config
import znicz_config
import logging
import error


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
        weights_amplitude: amplitude of the random distribution of weights.
        rand: rnd.Rand() object for initial weights generation.
        krn_: OpenCL kernel.
        s_activation: activation define for OpenCL source.
        weights_transposed: assume weights matrix as a transposed one.
    """
    def __init__(self, n_kernels=5, kx=5, ky=5, device=None,
                 weights_amplitude=None, rand=rnd.default,
                 weights_transposed=False):
        super(Conv, self).__init__(device=device)
        self.input = None
        self.output = formats.Vector()
        self.weights = formats.Vector()
        self.bias = formats.Vector()
        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.weights_amplitude = weights_amplitude
        self.rand = rand
        self.s_activation = "ACTIVATION_LINEAR"
        self.weights_transposed = weights_transposed
        self.exports.extend(("s_activation", "kx", "ky", "n_kernels"))

    def init_unpickled(self):
        super(Conv, self).init_unpickled()
        self.cl_sources_["conv.cl"] = {}
        self.krn_ = None

    def get_weights_amplitude(self):
        """
        Returns: weights amplitude for initial random distribution,
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

        if self.weights_amplitude == None:
            # Get weights amplitude and cap it to 0.05
            self.weights_amplitude = min(self.get_weights_amplitude(), 0.05)
        batch_size = self.input.v.shape[0]
        sy = self.input.v.shape[1]
        sx = self.input.v.shape[2]
        n_channels = self.input.v.size // (batch_size * sx * sy)
        n_weights = self.n_kernels * self.kx * self.ky * n_channels
        if self.weights.v == None or self.weights.v.size != n_weights:
            self.weights.reset()
            self.weights.v = numpy.zeros(n_weights, dtype=self.input.v.dtype)
            self.rand.fill(self.weights.v, -self.weights_amplitude,
                           self.weights_amplitude)
            self.weights.v = self.weights.v.reshape(self.n_kernels,
                self.kx * self.ky * n_channels)
            # Reshape weights as a matrix:
            if self.weights_transposed:
                a = self.weights.v.transpose().copy()
                self.weights.v.shape = a.shape
                self.weights.v[:] = a[:]
        if (self.bias.v == None or
            self.bias.v.size != self.n_kernels):
            self.bias.reset()
            self.bias.v = numpy.zeros(self.n_kernels, dtype=self.input.v.dtype)
            self.rand.fill(self.bias.v, -self.weights_amplitude,
                           self.weights_amplitude)

        if config.unit_test:
            batch_size <<= 1  # check for overflow
        output_size = batch_size * (self.n_kernels *
            (sx - self.kx + 1) * (sy - self.ky + 1))
        if self.output.v == None or self.output.v.size != output_size:
            self.output.reset()
            self.output.v = numpy.zeros([batch_size,
                sy - self.ky + 1, sx - self.kx + 1, self.n_kernels],
                dtype=self.input.v.dtype)
        del output_size

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if config.unit_test:
            batch_size >>= 1
            self.output.vv = self.output.v
            self.output.v = self.output.v[:batch_size]
            formats.assert_addr(self.output.v, self.output.vv)

        if self.device == None:
            return

        if self.krn_ == None:
            defines = {
                self.s_activation: 1,
                'BLOCK_SIZE': self.device.info.BLOCK_SIZE[config.c_dtype],
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
                config.cache_dir, sx, sy, n_channels, self.kx, self.ky,
                self.n_kernels))

            self.krn_ = pyopencl.Kernel(self.prg_, "feed_layer")
            self.krn_.set_arg(0, self.input.v_)
            self.krn_.set_arg(1, self.weights.v_)
            self.krn_.set_arg(2, self.output.v_)
            self.krn_.set_arg(3, self.bias.v_)

    def print_times(self, t_start):
        """Show some statistics.
        """
        log = self.log()
        if not log.isEnabledFor(logging.DEBUG):
            return
        self.output.map_read()
        y = self.output.v
        if y.dtype in (numpy.complex64, numpy.complex128):
            self.log().debug("%s: %d samples with %d weights in %.2f sec: "
                "y: min avg max: %.6f %.6f %.6f" %
                (self.__class__.__name__, y.shape[0],
                 self.weights.v.size, time.time() - t_start,
                 min(y.real.min(), y.imag.min()),
                 (numpy.average(y.real) + numpy.average(y.imag)) * 0.5,
                 max(y.real.max(), y.imag.max())))
        else:
            self.log().debug("%s: %d samples with %d weights in %.2f sec: "
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
        block_size = self.device.info.BLOCK_SIZE[config.c_dtype]
        global_size = [formats.roundup(self.n_kernels, block_size),
                       formats.roundup(self.output.v.size // self.n_kernels,
                                       block_size)]
        local_size = [block_size, block_size]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_,
                                                 global_size, local_size)
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

    def get_weights_amplitude(self):
        """
        Returns: weights amplitude for initial random distribution,
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
