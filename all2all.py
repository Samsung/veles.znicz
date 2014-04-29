"""
Created on Mar 20, 2013

All2All units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import time

import veles.config as config
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units


class All2All(nn_units.Forward):
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
        output_shape: shape of the output layer (may be Vector).
        krn_: OpenCL kernel.
        s_activation: activation define for OpenCL source.
        weights_transposed: assume weights matrix as a transposed one.

        weights_filling: rand weight filling
                         ("uniform" (default) or "gaussian")
        weights_magnitude: magnitude of uniform weight distribution.
        weights_stddev: StdDev of normal weight distributtion
    """
    def __init__(self, workflow, **kwargs):
        output_shape = kwargs.get("output_shape")
        kwargs["output_shape"] = output_shape

        weights_filling = kwargs.get("weights_filling", "uniform")
        weights_magnitude = kwargs.get("weights_magnitude", None)
        weights_stddev = kwargs.get("weights_stddev", 0.05)

        kwargs["weights_magnitude"] = weights_magnitude
        kwargs["weights_filling"] = weights_filling
        kwargs["weights_stddev"] = weights_stddev

        super(All2All, self).__init__(workflow, **kwargs)
        self.input = None
        self.weights_filling = weights_filling
        self.weights_stddev = weights_stddev
        self.weights_magnitude = weights_magnitude
        self.output = formats.Vector()
        self.weights = formats.Vector()
        self.bias = formats.Vector()
        self.output_shape = output_shape
        self.s_activation = "ACTIVATION_LINEAR"
        self.exports.append("s_activation")

    def init_unpickled(self):
        super(All2All, self).init_unpickled()
        self.cl_sources_["forward.cl"] = {}
        self.krn_ = None

    def get_weights_magnitude(self):
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

    def initialize(self, device, **kwargs):
        super(All2All, self).initialize(device=device, **kwargs)

        if self.weights_magnitude is None:
            # Get weights magnitude and cap it to 0.05
            self.weights_magnitude = min(self.get_weights_magnitude(), 0.05)
        output_shape = (self.output_shape.v.shape[1:]
                        if isinstance(self.output_shape, formats.Vector)
                        else self.output_shape)
        output_size = int(numpy.prod(output_shape))
        n_weights = (self.input.v.size // self.input.v.shape[0] * output_size)
        if self.weights.v is None or self.weights.v.size != n_weights:
            self.weights.reset()
            self.weights.v = numpy.zeros(n_weights, dtype=self.input.v.dtype)
            if self.weights_filling == "uniform":
                self.rand.fill(self.weights.v, -self.weights_magnitude,
                               self.weights_magnitude)
            elif self.weights_filling == "gaussian":
                self.rand.fill_normal_real(self.weights.v, 0,
                                           self.weights_stddev)
            else:
                assert False
            self.weights.v = self.weights.v.reshape([
                output_size, self.input.v.size // self.input.v.shape[0]])
            # Reshape weights as a matrix:
            if self.weights_transposed:
                a = self.weights.v.transpose().copy()
                self.weights.v.shape = a.shape
                self.weights.v[:] = a[:]
        if (self.bias.v is None or self.bias.v.size != output_size):
            self.bias.reset()
            self.bias.v = numpy.zeros(output_size, dtype=self.input.v.dtype)
            if self.weights_filling == "uniform":
                self.rand.fill(self.bias.v, -self.weights_magnitude,
                               self.weights_magnitude)
            elif self.weights_filling == "gaussian":
                self.rand.fill_normal_real(self.bias.v, 0, self.weights_stddev)
            else:
                assert False

        if (self.output.v is None or
                self.output.v.size != self.input.v.shape[0] * output_size):
            self.output.reset()
            self.output.v = numpy.zeros([self.input.v.shape[0], output_size],
                                        dtype=self.input.v.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if self.device is None:
            return

        if self.krn_ is None:
            defines = {
                self.s_activation: 1,
                'BLOCK_SIZE': self.device.device_info.BLOCK_SIZE[
                    opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)],
                'H': self.weights.v.size // output_size,
                'Y': output_size,
                'BATCH': self.output.v.shape[0]}
            if self.weights_transposed:
                defines['WEIGHTS_TRANSPOSED'] = 1
            self.build_program(defines, "%s/feed_%d_%d.cl" %
                               (config.root.common.cache_dir,
                                self.input.v.size // self.input.v.shape[0],
                                output_size),
                               dtype=self.input.v.dtype)

            self.krn_ = self.get_kernel("feed_layer")
            self.krn_.set_arg(0, self.input.v_)
            self.krn_.set_arg(1, self.weights.v_)
            self.krn_.set_arg(2, self.output.v_)
            self.krn_.set_arg(3, self.bias.v_)

        output_size = int(self.output.v.size // self.output.v.shape[0])
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)]
        self._global_size_ = [formats.roundup(output_size, block_size),
                              formats.roundup(self.output.v.shape[0],
                                              block_size)]
        self._local_size_ = [block_size, block_size]

    def print_times(self, t_start):
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
        self.output.unmap()
        self.input.unmap()
        self.weights.unmap()
        self.bias.unmap()

        self.execute_kernel(self.krn_, self._global_size_,
                            self._local_size_).wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.output.map_invalidate()
        self.input.map_read()
        self.weights.map_read()
        self.bias.map_read()
        a = formats.reshape(
            self.input.v, [self.input.v.shape[0],
                           self.input.v.size // self.input.v.shape[0]])
        b = self.weights.v
        if not self.weights_transposed:
            b = b.transpose()
        v = numpy.dot(a, b)
        v += self.bias.v
        self.output.v[:] = v[:]


class All2AllTanh(All2All):
    """All2All with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).
    """
    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_TANH"
        super(All2AllTanh, self).initialize(device=device, **kwargs)
        self.output.supposed_maxvle = 1.7159

    def get_weights_magnitude(self):
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            return (1.0 / (self.input.supposed_maxvle * 0.6666) /
                    (self.input.v.size // self.input.v.shape[0]))
        return (9.0 / (self.input.supposed_maxvle * 0.6666) /
                (self.input.v.size // self.input.v.shape[0]))

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllTanh, self).cpu_run()
        self.output.map_write()
        v = self.output.v.copy()
        v *= 0.6666
        numpy.tanh(v, v)
        v *= 1.7159
        self.output.v[:] = v[:]


class All2AllRELU(All2All):
    """All2All with RELU activation f(x) = log(1.0 + exp(x)).
    """
    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_RELU"
        super(All2AllRELU, self).initialize(device=device, **kwargs)
        self.output.supposed_maxvle = 10

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllRELU, self).cpu_run()
        self.output.map_write()
        v = self.output.v.copy()
        self.output.v[:] = numpy.where(v > 15, v,
                                       numpy.log(numpy.exp(v) + 1.0))


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
    def __init__(self, workflow, **kwargs):
        super(All2AllSoftmax, self).__init__(workflow, **kwargs)
        self.max_idx = formats.Vector()

    def init_unpickled(self):
        super(All2AllSoftmax, self).init_unpickled()
        self.krn_sm_ = None

    def initialize(self, device, **kwargs):
        # Always use 32-bit signed integers for output
        itype = opencl_types.numpy_dtype_to_opencl(numpy.int32)
        self.cl_sources_["softmax.cl"] = {"itype": itype}
        super(All2AllSoftmax, self).initialize(device=device, **kwargs)

        if (self.max_idx.v is None or
                self.max_idx.v.size != self.output.v.shape[0]):
            self.max_idx.v = numpy.zeros(self.output.v.shape[0],
                                         dtype=opencl_types.itypes[itype])
            self.max_idx.v_ = None

        self.max_idx.initialize(self.device)

        if self.device is None:
            return

        self.krn_sm_ = self.get_kernel("apply_exp")
        self.krn_sm_.set_arg(0, self.output.v_)
        self.krn_sm_.set_arg(1, self.max_idx.v_)

    def cpu_apply_exp(self):
        self.output.map_write()
        self.max_idx.map_invalidate()
        for i in range(0, self.output.v.shape[0]):
            sample = self.output[i]
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
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)]
        global_size = [self.output.v.shape[0] * block_size]
        local_size = [block_size]
        event = self.execute_kernel(self.krn_sm_, global_size, local_size)
        event.wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllSoftmax, self).cpu_run()
        self.cpu_apply_exp()

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        super(All2AllSoftmax, self).ocl_run()
        self.gpu_apply_exp()
