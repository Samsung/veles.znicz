# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 15, 2013

Gradient descent units for **all-to-all** perceptron units with different \
activations.

* :class:`GradientDescent` couples with :class:`veles.znicz.all2all.All2All`
* :class:`GDTanh` couples with :class:`veles.znicz.all2all.All2AllTanh`
* :class:`GDSM` couples with :class:`veles.znicz.all2all.All2AllSoftmax`
* :class:`GDRELU` couples with :class:`veles.znicz.all2all.All2AllRELU`\
    (NB: this ReLU is the smooth one from *Krizhevsky, Hinton et al.*,\
    not the strict one from CAFFE)

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

from veles.memory import reshape, Array
from veles.accelerated_units import IOpenCLUnit, ICUDAUnit, INumpyUnit
import veles.ocl_blas as ocl_blas
import veles.znicz.nn_units as nn_units
from collections import namedtuple


FastGDObjects = namedtuple("FastGDObjects", ("learning_rate",
                                             "weights", "bias"))
AdaDeltaGDObjects = namedtuple("AdaDeltaGDObjects", ("momentum",
                                                     "weights",
                                                     "gweights",
                                                     "bias",
                                                     "gbias",
                                                     "adom", "epsilon"))
AdaGradGDObjects = namedtuple("AdaGradGDObjects", ("epsilon",
                                                   "weights",
                                                   "bias"))


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class GradientDescent(nn_units.GradientDescentBase):
    """Gradient Descent unit for :class:`veles.znicz.all2all.All2All`.

    Attributes:
        output: assign before `initialize`!
        input: assign before `initialize`!
        err_output: assign before `initialize`!
        weights: assign before `initialize`!
        bias: assign before `initialize`!
        batch_size: assign before `initialize`!

        err_input: updates after `run`
        err_outpur: updates after `run`
        weights: updates after `run`
        bias: updates after `run`

        err_input: **creates** within `initialize`

    Attributes:
        krn_err_input_: OpenCL kernel for matrix multiplication.
        krn_weights_: OpenCL kernel for weights update.
        krn_err_output_: OpenCL kernel for err_output update.
        krn_bias_: OpenCL kernel for bias update.


        self.variant_gradient - variant of the method using a gradient
        0- old (gradient ->l1l2->add moment->new moment->upd weights)
        1- new ( gradient-> new moment->l1l2->
            add( adadelta adagard,fast and t.d.)->upd weights)
        2- TODO : NESTEROV
        3- Sparsing (different ways)
        self.variant_moment_gradient -
            variant of the method using a moment gradient
        gradient_weights_with_moment  -not the correct name
        may be gradient_weights_with_moment
        self.last_minibatch - need of loader
     """
    MAPPING = {"all2all"}
    SOLVERS = ("momentum", "adagrad", "adadelta", "fast")

    @property
    def solvers(self):
        return self._solvers

    @solvers.setter
    def solvers(self, arr):
        if "adagrad" in arr and "adadelta" in arr:
            raise ValueError("This solver is not have adagrad and adadelta")
        solvers = set()
        for value in arr:
            if value not in self.SOLVERS:
                raise ValueError(
                    "This solver is not supported: %s. Select one of %s.",
                    value, ", ".join(self.SOLVERS))
            solvers.add(value)
        self._solvers.clear()
        self._solvers.update(solvers)

    def __init__(self, workflow, **kwargs):
        self._solvers = set()
        super(GradientDescent, self).__init__(workflow, **kwargs)
        s = kwargs.get("solvers", set())
        self.solvers = s

        self.reduce_size = self.REDUCE_SIZE
        self.krn_err_input_ = None
        self.krn_weights_ = None
        self.krn_err_output_ = None
        self.krn_bias_ = None
        self.krn_compute_col_sums_ = None
        self.krn_err_output_name = None
        self.demand("weights")
        if self.include_bias:
            self.demand("bias")

        self.last_minibatch = None

        self.variant_gradient = kwargs.get("variant_gradient", True)
        self.variant_moment_gradient = (
            kwargs.get("variant_moment_gradient", True))
        if "fast" in self.solvers:
            self.fast = FastGDObjects(kwargs.get("fast_learning_rate", 0.02),
                                      Array(), Array())
        if "adadelta" in self.solvers:
            self.adadelta = AdaDeltaGDObjects(
                kwargs.get("adadelta_momentum", 0.9),
                Array(), Array(),
                Array(), Array(),
                kwargs.get("adadelta_adom", 0.3),
                kwargs.get("adadelta_epsilon", 1e-8))
            self.adadelta_adom = self.adadelta.adom

        if "adagrad" in self.solvers:
            self.adagrad = AdaGradGDObjects(
                kwargs.get("adagrad_epsilon", 1e-8),
                Array(), Array())

        self.last_minibatch = kwargs.get("last_minibatch", False)

    def initialize(self, device, **kwargs):
        if not self.input:
            return True

        super(GradientDescent, self).initialize(device=device, **kwargs)

        if "adadelta" in self.solvers:
            for vec in (self.adadelta.weights, self.adadelta.gweights):
                vec.reset(numpy.zeros_like(self.weights.mem))
            for vec in (self.adadelta.bias, self.adadelta.gbias):
                vec.reset(numpy.zeros_like(self.bias.mem))

        if "fast" in self.solvers:
            self.fast.bias.reset(numpy.zeros_like(self.bias.mem))
            self.fast.weights.reset(numpy.zeros_like(self.weights.mem))

        if "adagrad" in self.solvers:
            self.adagrad.bias.reset(numpy.zeros_like(self.bias.mem))
            self.adagrad.weights.reset(numpy.zeros_like(self.weights.mem))

        if "fast" in self.solvers:
            self.init_vectors(self.fast.weights, self.fast.bias)

        if "adadelta" in self.solvers:
            self.init_vectors(
                self.adadelta.weights,
                self.adadelta.gweights,
                self.adadelta.bias,
                self.adadelta.gbias)

        if "adagrad" in self.solvers:
            self.init_vectors(self.adagrad.weights, self.adagrad.bias)

        if (any(s in self.solvers for s in ("fast", "adagrad", "adadelta")) and
                not self.gradient_weights_with_moment):
            raise ValueError("Some of the solvers need moment vectors")

    def _gpu_init(self, blas_class):
        dtype = self.err_output.dtype

        self.gemm_ = blas_class.gemm(dtype)
        self.np_alpha = numpy.ones(1, dtype=dtype)
        self.np_beta = numpy.zeros(1, dtype=dtype)

        # The following code is for computing gradient for weight and bias
        if not self.need_gradient_weights:
            return

        self._weights_const = numpy.zeros(16, dtype=dtype)
        self._bias_const = numpy.zeros(16, dtype=dtype)

        side = self.weights_shape[0]
        other = self.weights.size // side
        assert side == self.err_output.sample_size
        assert other == self.input.sample_size
        batch = self.input.shape[0]
        defines = {
            "H": other,
            "Y": side,
            "BATCH": batch,
            "APPLY_GRADIENT": int(self.apply_gradient),
            "ACCUMULATE_GRADIENT": int(self.accumulate_gradient),
            "WEIGHTS_TRANSPOSED": int(self.weights_transposed),
            "REDUCE_SIZE": self.reduce_size
        }

        self.sources_["all2all/gradient_descent/weights_update"] = {
            "USE_ORTHO": int(bool(self.factor_ortho)),
            "USE_MOMENT": int(bool(self.gradient_weights_with_moment))
        }

        self.sources_["all2all/gradient_descent/bias_update"] = {
            "BIAS_SIZE": side,
            "OUTPUT_SIZE": batch,
            "USE_MOMENT": int(bool(self.gradient_bias_with_moment))
        }

        self.build_program(defines, "%s_%d_%d_%d" % (
            self.__class__.__name__, self.input.shape[0],
            self.input.sample_size, self.err_output.sample_size),
            dtype=dtype)

        self.krn_weights_ = self.get_kernel("weights_update")
        self.krn_weights_.set_args(self.weights.devmem,
                                   self.gradient_weights.devmem,
                                   self.accumulated_gradient_weights.devmem,
                                   self.gradient_weights_with_moment.devmem)

        if self.include_bias:
            self.krn_bias_ = self.get_kernel("bias_update")
            self.krn_bias_.set_args(
                self.err_output.devmem, self.bias.devmem,
                self.gradient_bias.devmem,
                self.accumulated_gradient_bias.devmem,
                self.gradient_bias_with_moment.devmem)

        if self.factor_ortho:
            self.krn_compute_col_sums_ = self.get_kernel("compute_col_sums")
            self.krn_compute_col_sums_.set_args(self.weights.devmem,
                                                self.col_sums.devmem)
            self.krn_weights_.set_arg(13, self.col_sums.devmem)

    def ocl_init(self):
        ocl_blas.OCLBLAS.attach_to_device(self.device)
        self._gpu_init(ocl_blas.OCLBLAS)

        if not self.need_gradient_weights:
            return

        side = self.weights_shape[0]
        other = self.weights.size // side

        self._global_size_weights = (self.weights.size,)
        self._local_size_weights = None

        if self.include_bias:
            self._global_size_bias = (side * self.reduce_size,)
            self._local_size_bias = (self.reduce_size,)

        self._global_size_ortho = (other * self.reduce_size,)
        self._local_size_ortho = (self.reduce_size,)

    def cuda_init(self):
        self._gpu_init(cublas.CUBLAS)

        if not self.need_gradient_weights:
            return

        side = self.weights_shape[0]
        other = self.weights.size // side

        block_size = self.device.suggest_block_size(self.krn_weights_)
        self._global_size_weights = (int(numpy.ceil(
            self.weights.size / block_size)), 1, 1)
        self._local_size_weights = (block_size, 1, 1)

        if self.include_bias:
            self._global_size_bias = (side, 1, 1)
            self._local_size_bias = (self.reduce_size, 1, 1)

        self._global_size_ortho = (other, 1, 1)
        self._local_size_ortho = (self.reduce_size, 1, 1)

    def moment_use(self, gradient_w_moment, grad):
        if gradient_w_moment:
            if self.variant_moment_gradient:
                gradients = (grad +
                             gradient_w_moment.mem * self.gradient_moment)
            else:
                gradients = (
                    (1 - self.gradient_moment) * grad +
                    gradient_w_moment.mem * self.gradient_moment)
            gradient_w_moment.mem[:] = gradients[:]
        else:
            gradients = grad
        return gradients

    def apply_gradient_f(self, gradient, vec, transposed):
        if self.apply_gradient:
            vec.mem += gradient

    def numpy_update(self, s):
        f_ortho_use = False if s == 'bias' else self.factor_ortho

        if s == 'weights':
            self.gradient_weights.map_read()
            for vec in (self.weights,
                        self.accumulated_gradient_weights,
                        self.gradient_weights_with_moment):
                vec.map_write()
            v_trans = getattr(self, s + "_transposed")
        elif s == 'bias':
            self.gradient_bias.map_read()
            for vec in (self.bias,
                        self.accumulated_gradient_bias,
                        self.gradient_bias_with_moment):
                vec.map_write()
            v_trans = False

        vec = getattr(self, s)
        grad_vec = getattr(self, "gradient_" + s)
        acc_vec = getattr(self, "accumulated_gradient_" + s)
        vec_old = getattr(self, "gradient_%s_with_moment" % s)
        if "fast" in self.solvers:
            f_vec = getattr(self.fast, s)
        if "adagrad" in self.solvers:
            adagard_vec = getattr(self.adagrad, s)
        if "adadelta" in self.solvers:
            adadelta_vec = getattr(self.adadelta, s)
            adadelta_gvec = getattr(self.adadelta, "g" + s)

        lr = self.learning_rate
        factor_l12 = self.weights_decay
        l1_vs_l2 = self.l1_vs_l2

        if self.variant_gradient:
            gradient = -nn_units.GradientDescentBase.numpy_gradient_step(
                vec.mem, grad_vec.mem, lr, factor_l12, l1_vs_l2, f_ortho_use,
                v_trans)
            gradient = self.accumulate_gradient_f(acc_vec, gradient)
            # if "momentum" in self.solvers:
            gradient = self.moment_use(vec_old, gradient)
        else:
            # it is RNN
            gradient = self.accumulate_gradient_f(acc_vec, grad_vec)

            gradient = self.moment_use(vec_old, gradient)
            gradient = -nn_units.GradientDescentBase.numpy_gradient_step(
                vec.mem, gradient, lr, factor_l12, l1_vs_l2, f_ortho_use,
                v_trans)
        if "adagrad" in self.solvers:
            gradient = self.apply_adagrad(adagard_vec, vec_old, gradient)
        if "adadelta" in self.solvers:
            gradient = self.apply_adadelta(adadelta_vec, adadelta_gvec,
                                           vec_old, gradient)
        if "fast" in self.solvers:
            self.apply_fast(f_vec, vec_old)

        self.apply_gradient_f(gradient, vec, v_trans)

        if "fast" in self.solvers and self.apply_gradient and not v_trans:
            vec.mem -= f_vec.mem

    def apply_fast(self, f_vec, vec_old):
        f_vec.mem *= 0.95
        f_vec.mem[:] = f_vec + self.fast.learning_rate * vec_old.mem

    def apply_adagrad(self, adagard_vec, vec_old, gradient):
        adagard_vec.map_write()
        adagard_vec.mem += (vec_old.mem ** 2)
        adagard_vec.map_read()
        gradient *= numpy.sqrt(adagard_vec.mem + self.adagrad.epsilon)

        return gradient

    def apply_adadelta(self, adadelta_vec, adadelta_gvec, vec_old, gradient):
        adadelta_vec.map_write()
        adadelta_gvec.map_write()
        adadelta_gvec.mem = (self.adadelta.adom * adadelta_gvec.mem +
                             (1 - self.adadelta.adom) * vec_old.mem ** 2)
        s1, s2 = (numpy.sqrt(m.mem + self.adadelta.epsilon)
                  for m in (adadelta_vec, adadelta_gvec))
        gradient *= s1 / s2
        adadelta_vec.mem = (self.adadelta_adom * adadelta_vec.mem +
                            (1 - self.adadelta_adom) * gradient ** 2)
        self.adadelta_adom = 0 if (
            self.last_minibatch) else self.adadelta.momentum
        return gradient

    def numpy_weights_update(self):
        if not self.need_gradient_weights:
            return
        self.input.map_read()
        self.output.map_read()
        self.err_output.map_write()

        err_output = reshape(
            self.err_output.mem,
            [self.err_output.shape[0], self.err_output.sample_size])

        inp = reshape(
            self.input.mem, [self.input.shape[0], self.input.sample_size])

        self.gradient_weights.map_write()
        if self.weights_transposed:
            numpy.dot(inp.transpose(), err_output, self.gradient_weights.mem)
        else:
            numpy.dot(err_output.transpose(), inp, self.gradient_weights.mem)

        self.numpy_update('weights')

    def numpy_bias_update(self):
        if not self.need_gradient_weights or not self.include_bias:
            return
        self.err_output.map_read()

        self.gradient_bias.map_write()
        self.gradient_bias.mem[:] = self.err_output.mem.sum(axis=0)

        self.numpy_update('bias')

    def numpy_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return
        self.err_input.map_invalidate()
        self.err_output.map_read()
        self.weights.map_read()
        err_output = reshape(
            self.err_output.mem,
            [self.err_output.shape[0], self.err_output.sample_size])
        err_input = reshape(
            self.err_input.mem,
            [self.err_input.shape[0], self.err_input.sample_size])
        if self.weights_transposed:
            bp = numpy.dot(err_output, self.weights.mem.transpose())
        else:
            bp = numpy.dot(err_output, self.weights.mem)
        bp *= self.err_input_alpha
        err_input *= self.err_input_beta
        err_input += bp

    def numpy_run(self):
        """Do gradient descent.
        """
        self.numpy_err_output_update()
        self.numpy_err_input_update()
        self.numpy_weights_update()
        self.numpy_bias_update()
        self.print_debug_data()

    def ocl_run(self):
        """Do gradient descent.
        """
        self.gpu_err_output_update()
        # TODO(a.kazantsev): remove intel_opencl_workaround flag.
        if self.intel_opencl_workaround:
            self.numpy_err_input_update()
        else:
            self.gpu_err_input_update()
        self.gpu_weights_update()
        self.gpu_bias_update()
        self.print_debug_data()

    def cuda_run(self):
        """Do gradient descent.
        """
        self.gpu_err_output_update()
        self.gpu_err_input_update()
        self.gpu_weights_update()
        self.gpu_bias_update()
        self.print_debug_data()

    def gpu_err_input_update(self):
        if not self.need_err_input:
            return

        self.unmap_vectors(self.err_output, self.weights, self.err_input)

        self.np_alpha[0] = self.err_input_alpha
        self.np_beta[0] = self.err_input_beta

        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_T
            if self.weights_transposed else cublas.CUBLAS_OP_N,
            cublas.CUBLAS_OP_N,
            self.err_input.sample_size, self.err_output.shape[0],
            self.err_output.sample_size,
            self.np_alpha, self.weights.devmem, self.err_output.devmem,
            self.np_beta, self.err_input.devmem)

    def gpu_weights_update(self):
        if not self.need_gradient_weights:
            return

        self.unmap_vectors(self.err_output, self.gradient_weights, self.input)

        self.np_alpha[0] = 1.0
        self.np_beta[0] = 0.0

        if self.weights_transposed:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.err_output.sample_size, self.input.sample_size,
                self.err_output.shape[0],
                self.np_alpha, self.err_output.devmem, self.input.devmem,
                self.np_beta, self.gradient_weights.devmem)
        else:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.input.sample_size, self.err_output.sample_size,
                self.err_output.shape[0],
                self.np_alpha, self.input.devmem, self.err_output.devmem,
                self.np_beta, self.gradient_weights.devmem)

        # Accumulate/apply gradient
        super(GradientDescent, self).gpu_weights_update()


class GDSoftmax(GradientDescent):
    """Gradient Descent for :class:`veles.znicz.all2all.All2AllSoftmax`.

    We minimize cross-entropy error function for softmax, so gradient descent
    is the same as in :class:`veles.znicz.gd.GradientDescent`.
    """
    MAPPING = {"softmax"}


class GDTanh(nn_units.GradientDescentWithActivation, GradientDescent):
    """Gradient Descent for

    :math:`f(x) = 1.7159 \\tanh(0.6666  s), s = (W  x + b)`,
        :math:`y = a \cdot \\tanh(b s)`

    :math:`f'(s) = (a \\cdot \\tanh(b  s))' = a \\cdot \\tanh'(b  s) \\cdot b`

    :math:`= a (1 - \\tanh^2(b s)) * b  =  a b - a * b * \\tanh^2(b s)`

    :math:`= a b - y * y * b / a =  y^2 (-b / a) + (a \\cdot b)`

    :math:`z = y^2 (-0.388484177) + 1.14381894`
    """

    MAPPING = {"all2all_tanh"}

    def numpy_err_output_update(self):
        """Multiply err_output by activation derivative
        by s in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= output * output * (-0.388484177) + 1.14381894

    def initialize(self, device, **kwargs):
        self.sources_["gradient_descent_tanh"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        return super(GDTanh, self).initialize(device=device, **kwargs)


class GDRELU(nn_units.GradientDescentWithActivation, GradientDescent):
    """
    Gradient Descent for :math:`f(x) = \\log(1 + \\exp(s))`

    :math:`s = (W x + b)`

    :math:`y = \\log(1.0 + \\exp(s))`

    :math:`f'(s) = \\frac{1}{1 + \\exp(-s)} = 1 - \\exp(-y)`
    """

    MAPPING = {"all2all_relu"}

    def numpy_err_output_update(self):
        """Multiply err_output by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= 1.0 - numpy.exp(-output)

    def initialize(self, device, **kwargs):
        self.sources_["gradient_descent_relu"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        return super(GDRELU, self).initialize(device=device, **kwargs)


class GDStrictRELU(nn_units.GradientDescentWithActivation, GradientDescent):
    """Gradient Descent for strict ReLU (like in CAFFE)

    :math:`f(x) = \\max(x, 0)`

    :math:`f'(s) = \\begin{cases}1 & s > 0 \\\\ 0 & else. \\\\ \\end{cases}`
    """

    MAPPING = {"all2all_str"}

    def numpy_err_output_update(self):
        """Multiply err_output by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= numpy.greater(output, 0)

    def initialize(self, device, **kwargs):
        self.sources_["gradient_descent_strict_relu"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        return super(GDStrictRELU, self).initialize(device=device, **kwargs)


class GDSigmoid(nn_units.GradientDescentWithActivation, GradientDescent):
    """Gradient Descent for Sigmoid activation.
    """

    MAPPING = {"all2all_sigmoid"}

    def numpy_err_output_update(self):
        """Multiply err_output by activation derivative
        by s in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= output * (1.0 - output)

    def initialize(self, device, **kwargs):
        self.sources_["gradient_descent_sigmoid"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        return super(GDSigmoid, self).initialize(device=device, **kwargs)
