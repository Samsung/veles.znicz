"""
Created on Mar 20, 2013

All-to-all perceptoron layers: simple (:class:`All2All`) and with \
activation function (:class:`All2AllTanh`, :class:`All2AllRELU` and  \
:class:`All2AllSoftmax`).


Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division
import cuda4py.blas as cublas
import numpy
from zope.interface import implementer

from veles.accelerated_units import IOpenCLUnit
import veles.error as error
from veles.memory import reshape, roundup, Vector
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
class All2All(nn_units.NNLayerBase):
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

    MAPPING = {"all2all"}

    def __init__(self, workflow, **kwargs):
        super(All2All, self).__init__(workflow, **kwargs)
        self.output_shape = kwargs["output_shape"]
        self.output_shape = (
            [self.output_shape] if isinstance(self.output_shape, int)
            else list(self.output_shape))
        self.s_activation = "ACTIVATION_LINEAR"
        self.exports.append("s_activation")
        self._global_size = None
        self._local_size = None
        self.demand("input")

    def init_unpickled(self):
        super(All2All, self).init_unpickled()
        self.cl_sources_["all2all/forward"] = {}

    def get_weights_magnitude(self):
        """
        Returns: weights range magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        vle = (1.0 / self.input.max_supposed /
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
                        if isinstance(self.output_shape, Vector)
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
        if (self.include_bias and (self.bias.mem is None or
                                   self.bias.mem.size != output_size)):
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
            self.output.mem = numpy.zeros(
                [self.input.mem.shape[0]] + output_shape,
                dtype=self.input.mem.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        self.backend_init()

    def cuda_init(self):
        dtype = self.input.dtype
        self.gemm = (cublas.CUBLAS.sgemm if dtype == numpy.float32
                     else cublas.CUBLAS.dgemm)
        self.np_one = numpy.ones(1, dtype=dtype)
        self.np_zero = numpy.zeros(1, dtype=dtype)
        if self.weights_transposed:
            raise NotImplementedError("TODO(a.kazantsev): implement")
        else:
            self.A = self.weights.devmem
            self.B = self.input.devmem
            self.transA = cublas.CUBLAS_OP_T
            self.transB = cublas.CUBLAS_OP_N
            self.rowsCountA = self.weights.shape[0]
            self.columnCountB = self.input.shape[0]
        self.commonSideLength = self.input.sample_size
        self.build_program({"OUTPUT_SAMPLE_SIZE": self.output.sample_size,
                            self.s_activation: 1,
                            "INCLUDE_BIAS": int(self.include_bias),
                            "Y": self.output.sample_size},
                           "%s_%d_%d_%d" %
                           (self.__class__.__name__, self.input.shape[0],
                            self.input.sample_size, self.output.sample_size),
                           dtype=dtype)
        if self.include_bias:
            self.assign_kernel("apply_bias_with_activation")
            self.set_args(self.output, self.bias)
            self._global_size_bias = (self.output.sample_size,
                                      self.output.shape[0], 1)

    def ocl_init(self):
        output_shape = (self.output_shape.mem.shape[1:]
                        if isinstance(self.output_shape, Vector)
                        else self.output_shape)
        output_size = int(numpy.prod(output_shape))
        a_width = self.output.mem.shape[0]
        b_width = output_size
        ab_common = self.weights.mem.size // output_size

        block_size = self.device.device_info.get_block_size(
            kernel="matrix_multiplication", dtype=self.input.dtype)

        defines = {
            "BLOCK_SIZE": block_size,
            self.s_activation: 1,
            "WEIGHTS_TRANSPOSED": int(self.weights_transposed),
            "INCLUDE_BIAS": int(self.include_bias),
            "H": ab_common,
            "Y": b_width,
            "BATCH": a_width}

        self.build_program(defines, "%s_%d_%d_%d" %
                           (self.__class__.__name__,
                            self.input.shape[0],
                            self.input.sample_size,
                            output_size),
                           dtype=self.input.mem.dtype)

        self.assign_kernel("feed_layer")
        if self.include_bias:
            self.set_args(self.input, self.weights, self.bias, self.output)
        else:
            self.set_args(self.input, self.weights, self.output)

        self._global_size = [roundup(b_width, block_size),
                             roundup(a_width, block_size)]
        self._local_size = [block_size, block_size]

    def ocl_run(self):
        if self.prefer_numpy:
            return self.cpu_run()
        return super(All2All, self).ocl_run()

    def cuda_run(self):
        self.output.unmap()
        self.input.unmap()
        self.weights.unmap()
        self.bias.unmap()

        self.gemm(self.device.blas, self.transA, self.transB,
                  self.rowsCountA, self.columnCountB, self.commonSideLength,
                  self.np_one, self.A, self.B,
                  self.np_zero, self.output.devmem)

        if self.include_bias:
            self.execute_kernel(self._global_size_bias, (1, 1, 1))

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.output.map_invalidate()
        self.input.map_read()
        self.weights.map_read()
        self.bias.map_read()
        mem = numpy.dot(self.input.matrix,
                        self.weights.mem if self.weights_transposed
                        else self.weights.mem.transpose())
        if self.include_bias:
            mem += self.bias.mem
        reshape(self.output.mem, mem.shape)[:] = mem[:]


class All2AllTanh(All2All):
    """All2All with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).
    """
    A = 1.7159
    B = 0.6666
    C = 9.0  # tanh(C) -> 1
    MAPPING = {"all2all_tanh"}

    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_TANH"
        super(All2AllTanh, self).initialize(device=device, **kwargs)
        self.output.max_supposed = All2AllTanh.A

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllTanh, self).cpu_run()
        self.output.map_write()
        mem = self.output.mem
        mem *= All2AllTanh.B
        numpy.tanh(mem, mem)
        mem *= All2AllTanh.A


class All2AllRELU(All2All):
    """All2All with RELU activation f(x) = log(1.0 + exp(x)).
    """

    MAPPING = {"all2all_relu"}

    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_RELU"
        super(All2AllRELU, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 10

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllRELU, self).cpu_run()
        self.output.map_write()
        mem = self.output.mem
        mem[:] = numpy.where(mem > 15, mem, numpy.log(numpy.exp(mem) + 1.0))


class All2AllStrictRELU(All2All):
    """All2All with RELU activation f(x) = max(x, 0).
    """

    MAPPING = {"all2all_str"}

    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_STRICT_RELU"
        super(All2AllStrictRELU, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 10

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllRELU, self).cpu_run()
        self.output.map_write()
        mem = self.output.mem
        numpy.clip(mem, 0.0, 1.0e30, mem)


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

    MAPPING = {"softmax"}

    def __init__(self, workflow, **kwargs):
        super(All2AllSoftmax, self).__init__(workflow, **kwargs)
        self.max_idx = Vector()
        self.reduce_size = 64

    def init_unpickled(self):
        super(All2AllSoftmax, self).init_unpickled()
        self.krn_sm_ = None
        self._force_gpu_apply_exp = False

    def initialize(self, device, **kwargs):
        self.reduce_size = min(self.reduce_size,
                               int(numpy.prod(self.output_shape)))
        self.cl_sources_["all2all/softmax"] = {
            "REDUCE_SIZE": self.reduce_size
        }
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

        if self.device is not None:
            self.krn_sm_ = self.get_kernel("apply_exp")
            self.krn_sm_.set_args(self.output.devmem, self.max_idx.devmem)

    def cpu_apply_exp(self):
        self.output.map_write()
        self.max_idx.map_invalidate()
        out = self.output.mem
        out = reshape(out, (out.shape[0], out.size // out.shape[0]))
        for i, sample in enumerate(out):
            im = sample.argmax()
            self.max_idx[i] = im
            m = sample[im]
            sample -= m
            numpy.exp(sample, sample)
            smm = sample.sum()
            sample /= smm

    def ocl_apply_exp(self):
        self.output.unmap()
        self.max_idx.unmap()
        global_size = [self.output.shape[0] * self.reduce_size]
        local_size = [self.reduce_size]
        self.execute_kernel(global_size, local_size, self.krn_sm_)

    def cuda_apply_exp(self):
        self.output.unmap()
        self.max_idx.unmap()
        global_size = (self.output.shape[0], 1, 1)
        local_size = (self.reduce_size, 1, 1)
        self.execute_kernel(global_size, local_size, self.krn_sm_)

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllSoftmax, self).cpu_run()
        if not self._force_gpu_apply_exp:
            self.cpu_apply_exp()

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self._force_gpu_apply_exp = True
        super(All2AllSoftmax, self).ocl_run()
        self.ocl_apply_exp()

    def cuda_run(self):
        """Forward propagation from batch on GPU.
        """
        self._force_gpu_apply_exp = True
        super(All2AllSoftmax, self).cuda_run()
        self.cuda_apply_exp()
