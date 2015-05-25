# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mar 20, 2013

All-to-all perceptron layers: simple (:class:`All2All`) and with \
activation function (:class:`All2AllTanh`, :class:`All2AllRELU` and  \
:class:`All2AllSoftmax`).

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


from __future__ import division
import cuda4py.blas as cublas
import numpy
from zope.interface import implementer

from veles.accelerated_units import IOpenCLUnit, ICUDAUnit, INumpyUnit
import veles.error as error
from veles.memory import reshape, Array
import veles.ocl_blas as ocl_blas
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
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
        output_sample_shape: shape of the output layer (may be Array).
        output_samples_number: the number of samples in the output If it is
        None (the default), it is taken from input.
        output_dtype: the dtype of output. If it is None (the default),
        it is taken from input.
        activation_mode: activation type. It is passed as a definition directly
        to OpenCL/CUDA source code.
        weights_transposed: assume weights matrix as a transposed one,
                            NOTE: only access order will be affected,
                            not a shape.

        weights_filling: rand weight filling
                         ("uniform" (default) or "gaussian")
        weights_stddev: magnitude of uniform weight distribution.
        weights_stddev: StdDev of normal weight distributtion
    """

    MAPPING = {"all2all"}

    def __init__(self, workflow, **kwargs):
        super(All2All, self).__init__(workflow, **kwargs)
        self.output_sample_shape = kwargs.get("output_sample_shape", tuple())
        self.output_samples_number = kwargs.get("output_samples_number")
        self.output_dtype = kwargs.get("output_dtype")
        self.activation_mode = "ACTIVATION_LINEAR"
        self.exports.append("activation_mode")
        self._global_size = None
        self._local_size = None
        self.demand("input", "output_sample_shape", "output_samples_number")

    def init_unpickled(self):
        super(All2All, self).init_unpickled()
        self.sources_["all2all/forward"] = {}

    @property
    def output_sample_shape(self):
        return self._output_sample_shape

    @output_sample_shape.setter
    def output_sample_shape(self, value):
        assert not self.is_initialized, \
            "Cannot set output_sample_shape after initialize() was called"
        self._set_output_sample_shape(value)

    def _set_output_sample_shape(self, value):
        if isinstance(value, int):
            self._output_sample_shape = (value,)
        elif hasattr(value, "shape"):
            self._output_sample_shape = value.shape[1:]
        elif hasattr(value, "__iter__"):
            self._output_sample_shape = tuple(value)
        else:
            raise TypeError("Unsupported output_sample_shape type: %s" %
                            type(value))

    @property
    def output_samples_number(self):
        if self.input:
            return self.input.shape[0]
        return self._output_samples_number

    @output_samples_number.setter
    def output_samples_number(self, value):
        if value is not None and not isinstance(value, int):
            raise TypeError("output_samples_number must be an integer")
        self._output_samples_number = value

    @property
    def output_shape(self):
        return (self.output_samples_number,) + self.output_sample_shape

    @property
    def neurons_number(self):
        return int(numpy.prod(self.output_sample_shape))

    def get_weights_magnitude(self):
        """
        Returns: weights range magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        vle = (1.0 / self.input.max_supposed /
               numpy.sqrt(self.input.sample_size))
        if self.weights_filling == "gaussian":
            vle /= 3
        return vle

    def fill_array(self, filling, array, stddev):
        if filling == "uniform":
            self.rand.fill(array, -stddev, stddev)
        elif filling == "gaussian":
            self.rand.fill_normal_real(array, 0, stddev)
        elif filling == "constant":
            array[:] = stddev
        else:
            raise error.BadFormatError("Invalid filling type %s" % filling)

    def initialize(self, device, **kwargs):
        if not self.input:
            assert self.output_samples_number is not None, \
                "self.input is not initialized and output_samples_number was "\
                "not specified => unable to validate/create output"
            if not self.output:
                assert self.output_dtype is not None, \
                    "self.input is not initialized and output_dtype was " \
                    "not specified => unable to create output"
                self.output.reset(numpy.zeros(
                    self.output_shape, self.output_dtype))
            else:
                assert self.output.shape == self.output_shape
            return True

        super(All2All, self).initialize(device=device, **kwargs)

        if self.weights_stddev is None:
            self.weights_stddev = min(self.get_weights_magnitude(), 0.05)
        if self.bias_stddev is None:
            self.bias_stddev = self.weights_stddev

        # Check that weights vector was not assigned from the outside
        self.weights_shape = (self.neurons_number, self.input.sample_size)
        weights_shape_t = tuple(reversed(self.weights_shape))
        if not self.weights:
            self.weights.reset(numpy.zeros(self.weights_shape,
                                           dtype=self.input.dtype))
            self.fill_array(self.weights_filling, self.weights.mem,
                            self.weights_stddev)
            if self.weights_transposed:
                self.weights.shape = weights_shape_t
        else:
            assert (self.weights.shape == weights_shape_t if
                    self.weights_transposed else weights_shape_t)

        if self.include_bias:
            # Check that bias was not assigned from the outside
            if not self.bias:
                self.bias.reset(numpy.zeros(
                    self.neurons_number, self.input.dtype))
                self.fill_array(self.bias_filling, self.bias.mem,
                                self.bias_stddev)
            else:
                assert self.bias.size == self.neurons_number

        self._create_output()
        self.init_vectors(self.input, self.output, self.weights, self.bias)

    def _create_output(self):
        if self.output and self.output.shape == self.output_shape:
            return
        if not self.output:
            self.output.reset(numpy.zeros(self.output_shape, self.input.dtype))
        else:
            assert self.output.shape == self.output_shape

    def _gpu_init(self, blas_class):
        dtype = self.input.dtype
        self.gemm_ = blas_class.gemm(dtype)
        self.np_one = numpy.ones(1, dtype)
        self.np_zero = numpy.zeros(1, dtype)
        self._transA = (cublas.CUBLAS_OP_N if self.weights_transposed
                        else cublas.CUBLAS_OP_T)
        self._transB = cublas.CUBLAS_OP_N
        self._A_ = self.weights.devmem
        self._B_ = self.input.devmem
        self._rowsCountA = self.weights_shape[0]
        self._columnCountB = self.input.shape[0]
        self._commonSideLength = self.input.sample_size
        self.build_program({"BIAS_SIZE": self.output.sample_size,
                            "OUTPUT_SIZE": self.output.size,
                            self.activation_mode: 1,
                            "INCLUDE_BIAS": int(self.include_bias),
                            "Y": self.output.sample_size},
                           "%s_%d_%d_%d" %
                           (self.__class__.__name__, self.input.shape[0],
                            self.input.sample_size, self.output.sample_size),
                           dtype=dtype)
        if self.include_bias or self.activation_mode != "ACTIVATION_LINEAR":
            self.assign_kernel("apply_bias_with_activation")
            self.set_args(self.output, self.bias)

    def cuda_init(self):
        self._gpu_init(cublas.CUBLAS)
        if self._kernel_ is not None:
            block_size = self.device.suggest_block_size(self._kernel_)
            self._global_size_bias = (
                int(numpy.ceil(self.output.size / block_size)), 1, 1)
            self._local_size_bias = (block_size, 1, 1)

    def ocl_init(self):
        ocl_blas.OCLBLAS.attach_to_device(self.device)
        self._gpu_init(ocl_blas.OCLBLAS)
        if self._kernel_ is not None:
            self._global_size_bias = (self.output.size,)
            self._local_size_bias = None

    def _gpu_run(self):
        self.unmap_vectors(self.output, self.input, self.weights, self.bias)

        self.gemm_(
            self.device.blas, self._transA, self._transB,
            self._rowsCountA, self._columnCountB, self._commonSideLength,
            self.np_one, self._A_, self._B_,
            self.np_zero, self.output.devmem)

        if self.include_bias or self.activation_mode != "ACTIVATION_LINEAR":
            self.execute_kernel(self._global_size_bias, self._local_size_bias)

    def ocl_run(self):
        if self.intel_opencl_workaround:
            return self.numpy_run()
        return self._gpu_run()

    def cuda_run(self):
        return self._gpu_run()

    def numpy_run(self):
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
        self.activation_mode = "ACTIVATION_TANH"
        super(All2AllTanh, self).initialize(device=device, **kwargs)
        self.output.max_supposed = All2AllTanh.A

    def numpy_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllTanh, self).numpy_run()
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
        self.activation_mode = "ACTIVATION_RELU"
        super(All2AllRELU, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 10

    def numpy_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllRELU, self).numpy_run()
        self.output.map_write()
        mem = self.output.mem
        mem[:] = numpy.where(mem > 15, mem, numpy.log(numpy.exp(mem) + 1.0))


class All2AllStrictRELU(All2All):
    """All2All with RELU activation f(x) = max(x, 0).
    """

    MAPPING = {"all2all_str"}

    def initialize(self, device, **kwargs):
        self.activation_mode = "ACTIVATION_STRICT_RELU"
        super(All2AllStrictRELU, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 10

    def numpy_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllStrictRELU, self).numpy_run()
        self.output.map_write()
        mem = self.output.mem
        numpy.clip(mem, 0.0, 1.0e30, mem)


class All2AllSigmoid(All2All):
    """All2All with Sigmoid activation f(x) = 1 / (1 + exp(-x)).
    """
    MAPPING = {"all2all_sigmoid"}

    def initialize(self, device, **kwargs):
        self.activation_mode = "ACTIVATION_SIGMOID"
        super(All2AllSigmoid, self).initialize(device=device, **kwargs)
        self.output.supposed_max_value = 10

    def numpy_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllSigmoid, self).numpy_run()
        self.output.map_write()
        mem = self.output.mem
        # 1 / (1 + numpy.exp(-mem))
        numpy.exp(-mem, mem)
        numpy.reciprocal(mem + 1, mem)


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
        self.max_idx = Array()
        self.reduce_size = 256

    def init_unpickled(self):
        super(All2AllSoftmax, self).init_unpickled()
        self.krn_sm_ = None
        self._force_gpu_apply_exp = False

    def initialize(self, device, **kwargs):
        self.reduce_size = min(self.reduce_size,
                               int(numpy.prod(self.output_sample_shape)))
        self.sources_["all2all/softmax"] = {
            "REDUCE_SIZE": self.reduce_size
        }
        super(All2AllSoftmax, self).initialize(device=device, **kwargs)
        if self.output.mem.size // self.output.mem.shape[0] <= 1:
            raise error.BadFormatError(
                "Output sample size should be greater than 1 for SoftMax.")

        if not self.max_idx:
            self.max_idx.reset(numpy.zeros(self.output.shape[0],
                                           dtype=numpy.int32))
        self.max_idx.initialize(self.device)

    def numpy_apply_exp(self):
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
        self.unmap_vectors(self.output, self.max_idx)
        global_size = (self.output.shape[0] * self.reduce_size,)
        local_size = (self.reduce_size,)
        self.execute_kernel(global_size, local_size, self.krn_sm_)

    def cuda_apply_exp(self):
        self.unmap_vectors(self.output, self.max_idx)
        global_size = (self.output.shape[0], 1, 1)
        local_size = (self.reduce_size, 1, 1)
        self.execute_kernel(global_size, local_size, self.krn_sm_)

    def numpy_run(self):
        """Forward propagation from batch on CPU only.
        """
        super(All2AllSoftmax, self).numpy_run()
        if not self._force_gpu_apply_exp:
            self.numpy_apply_exp()

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

    def ocl_init(self):
        super(All2AllSoftmax, self).ocl_init()
        self.krn_sm_ = self.get_kernel("apply_exp")
        self.krn_sm_.set_args(self.output.devmem, self.max_idx.devmem)

    def cuda_init(self):
        super(All2AllSoftmax, self).cuda_init()
        self.krn_sm_ = self.get_kernel("apply_exp")
        self.krn_sm_.set_args(self.output.devmem, self.max_idx.devmem)
