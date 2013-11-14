"""
Created on Mar 20, 2013

All2All units.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import formats
import numpy
import pyopencl
import time
import rnd
import config
import logging


class All2All(units.OpenCLUnit):
    """All2All with linear activation f(x) = x.

    Should be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        weights
        bias
        output

    Attributes:
        input: input as batch of samples.
        output: output as batch of samples.
        weights: matrix of weights.
        bias: bias.
        output_shape: shape of the output layer.
        weights_amplitude: amplitude of the random distribution of weights.
        rand: rnd.Rand() object for initial weights generation.
        krn_: OpenCL kernel.
        s_activation: activation define for OpenCL source.
        weights_transposed: assume weights matrix as a transposed one.
    """
    def __init__(self, output_shape=None, device=None, weights_amplitude=None,
                 rand=rnd.default, weights_transposed=False):
        super(All2All, self).__init__(device=device)
        self.input = None  # formats.Vector(device)
        self.output = formats.Vector(device)
        self.weights = formats.Vector(device)
        self.bias = formats.Vector(device)
        self.output_shape = output_shape
        self.weights_amplitude = weights_amplitude
        self.rand = rand
        self.s_activation = "ACTIVATION_LINEAR"
        self.weights_transposed = weights_transposed
        self.exports = ["weights", "bias"]

    def init_unpickled(self):
        super(All2All, self).init_unpickled()
        self.cl_sources_["%s/forward.cl" % (config.cl_dir)] = ""
        self.krn_ = None

    def get_weights_amplitude(self):
        """
        Returns: weights amplitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            return (1.0 / self.input.supposed_maxvle /
                (self.input.v.size // self.input.v.shape[0]))
        return (9.0 / self.input.supposed_maxvle /
                (self.input.v.size // self.input.v.shape[0]))

    def initialize(self):
        super(All2All, self).initialize()

        if self.weights_amplitude == None:
            # Get weights amplitude and cap it to 0.05
            self.weights_amplitude = min(self.get_weights_amplitude(), 0.05)
        n_weights = (self.input.v.size // self.input.v.shape[0] *
                     numpy.prod(self.output_shape))
        if self.weights.v == None or self.weights.v.size != n_weights:
            self.weights.reset()
            self.weights.v = numpy.zeros(n_weights, dtype=self.input.v.dtype)
            self.rand.fill(self.weights.v, -self.weights_amplitude,
                           self.weights_amplitude)
            self.weights.v = self.weights.v.reshape([
                numpy.prod(self.output_shape),
                self.input.v.size // self.input.v.shape[0]])
            # Reshape weights as a matrix:
            if self.weights_transposed:
                a = self.weights.v.transpose().copy()
                self.weights.v.shape = a.shape
                self.weights.v[:] = a[:]
        if (self.bias.v == None or
            self.bias.v.size != numpy.prod(self.output_shape)):
            self.bias.reset()
            self.bias.v = numpy.zeros([numpy.prod(self.output_shape)],
                                      dtype=self.input.v.dtype)
            self.rand.fill(self.bias.v, -self.weights_amplitude,
                           self.weights_amplitude)

        output_size = self.input.v.shape[0] * numpy.prod(self.output_shape)
        if self.output.v == None or self.output.v.size != output_size:
            self.output.reset()
            self.output.v = numpy.zeros([self.input.v.shape[0],
                                        numpy.prod(self.output_shape)],
                                        dtype=self.input.v.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if self.device == None:
            return

        if self.krn_ == None:
            output_size = (self.output.v.size //
                           self.output.v.shape[0])
            defines = ("%s\n"
                       "%s\n"
                       "#define %s\n"
                       "#define BLOCK_SIZE %d\n"
                       "#define H %d\n"
                       "#define Y %d\n"
                       "#define BATCH %d\n\n" %
                       ("#define WEIGHTS_TRANSPOSED"
                        if self.weights_transposed else "",
                        config.cl_defines[config.c_dtype], self.s_activation,
                        self.device.info.BLOCK_SIZE[config.c_dtype],
                        self.weights.v.size // output_size, output_size,
                        self.output.v.shape[0]))
            self.build_program(defines, "%s/feed_%d_%d.cl" % (config.cache_dir,
                self.input.v.size // self.input.v.shape[0],
                self.output.v.size // self.output.v.shape[0]))

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
        self.output.unmap()
        self.input.unmap()
        self.weights.unmap()
        self.bias.unmap()
        output_size = int(self.output.v.size //
                          self.output.v.shape[0])
        block_size = self.device.info.BLOCK_SIZE[config.c_dtype]
        global_size = [formats.roundup(output_size, block_size),
                       formats.roundup(self.output.v.shape[0], block_size)]
        local_size = [block_size, block_size]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_,
                                                 global_size, local_size)
        event.wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.output.map_invalidate()
        self.input.map_read()
        self.weights.map_read()
        self.bias.map_read()
        a = formats.reshape(self.input.v,
            [self.input.v.shape[0],
             self.input.v.size // self.input.v.shape[0]])
        b = self.weights.v
        if not self.weights_transposed:
            b = b.transpose()
        v = numpy.dot(a, b)
        v += self.bias.v
        self.output.v[:] = v[:]

    def run(self):
        t1 = time.time()
        retval = super(All2All, self).run()
        if retval:
            return retval
        self.print_times(t1)


class All2AllTanh(All2All):
    """All2All with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).
    """
    def initialize(self):
        self.s_activation = "ACTIVATION_TANH"
        super(All2AllTanh, self).initialize()
        self.output.supposed_maxvle = 1.7159

    def get_weights_amplitude(self):
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            return (1.0 / (self.input.supposed_maxvle * 0.6666) /
                    (self.input.v.size // self.input.v.shape[0]))
        return (9.0 / (self.input.supposed_maxvle * 0.6666) /
                (self.input.v.size // self.input.v.shape[0]))

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        retval = super(All2AllTanh, self).cpu_run()
        if retval:
            return retval
        self.output.map_write()
        v = self.output.v.copy()
        v *= 0.6666
        numpy.tanh(v, v)
        v *= 1.7159
        self.output.v[:] = v[:]


class All2AllSoftmax(All2All):
    """All2All with linear activation and softmax normalization.

    Should be assigned before initialize():

    Updates after run():
        max_idx

    Creates within initialize():
        max_idx

    Attributes:
        krn_sm_: kernel for softmax activation calculation.
        max_idx: indexes of element with maximum value for each sample.
    """
    def __init__(self, output_shape=None, device=None, weights_amplitude=None,
                 rand=rnd.default, weights_transposed=False):
        super(All2AllSoftmax, self).__init__(
            output_shape=output_shape, device=device,
            weights_amplitude=weights_amplitude, rand=rand,
            weights_transposed=weights_transposed)
        self.max_idx = formats.Vector()

    def init_unpickled(self):
        super(All2AllSoftmax, self).init_unpickled()
        self.krn_sm_ = None

    def initialize(self):
        itype = config.get_itype_from_size(numpy.prod(self.output_shape))
        global this_dir
        self.cl_sources_["%s/softmax.cl" % (config.cl_dir)] = (
            "#define itype %s" % (itype))
        super(All2AllSoftmax, self).initialize()

        if (self.max_idx.v == None or
            self.max_idx.v.size != self.output.v.shape[0]):
            self.max_idx.v = numpy.zeros(self.output.v.shape[0],
                dtype=config.itypes[itype])
            self.max_idx.v_ = None

        self.max_idx.initialize(self.device)

        if self.device == None:
            return

        self.krn_sm_ = pyopencl.Kernel(self.prg_, "apply_exp")
        self.krn_sm_.set_arg(0, self.output.v_)
        self.krn_sm_.set_arg(1, self.max_idx.v_)

    def cpu_apply_exp(self):
        self.output.map_write()
        self.max_idx.map_invalidate()
        for i in range(0, self.output.v.shape[0]):
            sample = self.output.v[i]
            im = sample.argmax()
            self.max_idx[i] = im
            m = sample[im]
            sample -= m
            numpy.exp(sample, sample)
            smm = sample.sum()
            sample /= smm

    def gpu_apply_exp(self):
        self.output.unmap()
        self.max_idx.unmap()
        block_size = self.device.info.BLOCK_SIZE[config.c_dtype]
        global_size = [self.output.v.shape[0] * block_size]
        local_size = [block_size]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
            self.krn_sm_, global_size, local_size)
        event.wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        retval = super(All2AllSoftmax, self).cpu_run()
        if retval:
            return retval
        self.cpu_apply_exp()

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        retval = super(All2AllSoftmax, self).gpu_run()
        if retval:
            return retval
        self.gpu_apply_exp()
