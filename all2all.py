"""
Created on Mar 20, 2013

All2All units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import logging
import numpy
import time
from zope.interface import implementer

from veles.opencl_units import IOpenCLUnit

import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
class All2All(nn_units.Forward):
    """All2All with linear activation f(x) = x.

    Must be assigned before initialize():
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
        s_activation: activation define for OpenCL source.
        weights_transposed: assume weights matrix as a transposed one.

        weights_filling: rand weight filling
                         ("uniform" (default) or "gaussian")
        weights_stddev: magnitude of uniform weight distribution.
        weights_stddev: StdDev of normal weight distributtion
    """
    def __init__(self, workflow, **kwargs):
        output_shape = kwargs.get("output_shape")
        kwargs["output_shape"] = output_shape
        super(All2All, self).__init__(workflow, **kwargs)
        self.output_shape = output_shape
        self.s_activation = "ACTIVATION_LINEAR"
        self.exports.append("s_activation")

    def init_unpickled(self):
        super(All2All, self).init_unpickled()
        self.cl_sources_["forward.cl"] = {}

    def get_weights_magnitude(self):
        """
        Returns: weights range magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        if self.input.mem.dtype in (numpy.complex64, numpy.complex128):
            vle = (1.0 / self.input.supposed_maxvle /
                   numpy.sqrt(self.input.mem.size // self.input.mem.shape[0]))
        else:
            vle = (9.0 / self.input.supposed_maxvle /
                   numpy.sqrt(self.input.mem.size // self.input.mem.shape[0]))
        if self.weights_filling == "gaussian":
            vle /= 3
        return vle

    def initialize(self, device, **kwargs):
        super(All2All, self).initialize(device=device, **kwargs)

        if self.weights_stddev is None:
            self.weights_stddev = min(self.get_weights_magnitude(), 0.05)
        if self.bias_stddev is None:
            self.bias_stddev = self.weights_stddev

        output_shape = (self.output_shape.mem.shape[1:]
                        if isinstance(self.output_shape, formats.Vector)
                        else self.output_shape)
        output_size = int(numpy.prod(output_shape))
        n_weights = (self.input.mem.size //
                     self.input.mem.shape[0] * output_size)
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
            else:
                raise error.BadFormatError("Invalid weights filling type")
            self.weights.mem = self.weights.mem.reshape([
                output_size, self.input.mem.size // self.input.mem.shape[0]])
            # Reshape weights as a matrix:
            if self.weights_transposed:
                a = self.weights.mem.transpose().copy()
                self.weights.mem.shape = a.shape
                self.weights.mem[:] = a[:]
        if self.bias.mem is None or self.bias.mem.size != output_size:
            self.bias.reset()
            self.bias.mem = numpy.zeros(output_size,
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

        if (self.output.mem is None or
                self.output.mem.size != self.input.mem.shape[0] * output_size):
            self.output.reset()
            self.output.mem = numpy.zeros([self.input.mem.shape[0],
                                           output_size],
                                          dtype=self.input.mem.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if self.device is None:
            return

        defines = {
            self.s_activation: 1,
            'BLOCK_SIZE': self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.input.mem.dtype)],
            'H': self.weights.mem.size // output_size,
            'Y': output_size,
            'BATCH': self.output.mem.shape[0]}
        if self.weights_transposed:
            defines['WEIGHTS_TRANSPOSED'] = 1
        self.build_program(defines, "feed_%d_%d.cl" %
                           (self.input.mem.size // self.input.mem.shape[0],
                            output_size),
                           dtype=self.input.mem.dtype)

        self.assign_kernel("feed_layer")
        self.set_args(self.input, self.weights, self.output, self.bias)

        output_size = int(self.output.mem.size // self.output.mem.shape[0])
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.mem.dtype)]
        self._global_size_ = [formats.roundup(output_size, block_size),
                              formats.roundup(self.output.mem.shape[0],
                                              block_size)]
        self._local_size_ = [block_size, block_size]

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
        self.output.unmap()
        self.input.unmap()
        self.weights.unmap()
        self.bias.unmap()

        self.execute_kernel(self._global_size_, self._local_size_).wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.output.map_invalidate()
        self.input.map_read()
        self.weights.map_read()
        self.bias.map_read()
        a = formats.reshape(
            self.input.mem, [self.input.mem.shape[0],
                             self.input.mem.size // self.input.mem.shape[0]])
        b = self.weights.mem
        if not self.weights_transposed:
            b = b.transpose()
        mem = numpy.dot(a, b)
        mem += self.bias.mem
        self.output.mem[:] = mem[:]


class All2AllTanh(All2All):
    """All2All with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).
    """
    A = 1.7159
    B = 0.6666
    C = 9.0  # tanh(C) -> 1

    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_TANH"
        super(All2AllTanh, self).initialize(device=device, **kwargs)
        self.output.supposed_maxvle = All2AllTanh.A

    def get_weights_magnitude(self):
        if self.input.mem.dtype in (numpy.complex64, numpy.complex128):
            vle = (1.0 / (self.input.supposed_maxvle * All2AllTanh.B) /
                   numpy.sqrt(self.input.mem.size // self.input.mem.shape[0]))
        else:
            vle = (All2AllTanh.C /
                   (self.input.supposed_maxvle * All2AllTanh.B *
                    numpy.sqrt(self.input.mem.size //
                               self.input.mem.shape[0])))
        if self.weights_filling == "gaussian":
            vle /= 3
        return vle

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllTanh, self).cpu_run()
        self.output.map_write()
        mem = self.output.mem.copy()
        mem *= All2AllTanh.B
        numpy.tanh(mem, mem)
        mem *= All2AllTanh.A
        self.output.mem[:] = mem[:]


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
        mem = self.output.mem.copy()
        self.output.mem[:] = numpy.where(mem > 15, mem,
                                         numpy.log(numpy.exp(mem) + 1.0))


class All2AllSoftmax(All2All):
    """All2All with linear activation and softmax normalization.

    Must be assigned before initialize():

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
        self.reduce_size = 64

    def init_unpickled(self):
        super(All2AllSoftmax, self).init_unpickled()
        self.krn_sm_ = None

    def initialize(self, device, **kwargs):
        self.reduce_size = min(self.reduce_size,
                               int(numpy.prod(self.output_shape)))
        self.cl_sources_["softmax.cl"] = {"REDUCE_SIZE": self.reduce_size}
        super(All2AllSoftmax, self).initialize(device=device, **kwargs)
        if self.output.mem.size // self.output.mem.shape[0] <= 1:
            raise error.BadFormatError(
                "Output sample size should be greater than 1 for SoftMax.")

        if (self.max_idx.mem is None or
                self.max_idx.mem.size != self.output.mem.shape[0]):
            self.max_idx.mem = numpy.zeros(self.output.mem.shape[0],
                                           dtype=numpy.int32)
            self.max_idx.devmem = None

        self.max_idx.initialize(self.device)

        if self.device is None:
            return

        self.krn_sm_ = self.get_kernel("apply_exp")
        self.krn_sm_.set_args(self.output.devmem, self.max_idx.devmem)

    def cpu_apply_exp(self):
        self.output.map_write()
        self.max_idx.map_invalidate()
        for i in range(0, self.output.mem.shape[0]):
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
        global_size = [self.output.mem.shape[0] * self.reduce_size]
        local_size = [self.reduce_size]
        event = self.execute_kernel(global_size, local_size, self.krn_sm_)
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
