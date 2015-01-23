"""
Created on Apr 15, 2013

Gradient descent units for **all-to-all** perceptron units with different \
activations.

* :class:`GradientDescent` couples with :class:`veles.znicz.all2all.All2All`
* :class:`GDTanh` couples with :class:`veles.znicz.all2all.All2AllTanh`
* :class:`GDSM` couples with :class:`veles.znicz.all2all.All2AllSoftmax`
* :class:`GDRELU` couples with :class:`veles.znicz.all2all.All2AllRELU`\
    (NB: this ReLU is the smooth one from *Krizhevsky, Hinton et al.*,\
    not the strict one from CAFFE)


Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import cuda4py.blas as cublas
import numpy
import time
from zope.interface import implementer

from veles.memory import reshape, roundup
from veles.accelerated_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
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
    """

    MAPPING = {"all2all"}

    def __init__(self, workflow, **kwargs):
        super(GradientDescent, self).__init__(workflow, **kwargs)
        self.reduce_size = 64
        self.cl_const = None
        self.krn_err_input_ = None
        self.krn_weights_ = None
        self.krn_err_output_ = None
        self.krn_bias_ = None
        self.krn_compute_col_sums_ = None
        self.krn_err_output_name = None

    def _gpu_init(self, defines):
        dtype = self.err_output.mem.dtype
        self.cl_const = numpy.zeros(9, dtype=dtype)

        side = self.weights.shape[1 if self.weights_transposed else 0]
        other = self.weights.size // side
        assert side == self.err_output.sample_size
        assert other == self.input.sample_size
        batch = self.input.mem.shape[0]
        defines.update({
            "H": other,
            "Y": side,
            "BATCH": batch,
            "APPLY_GRADIENT": int(self.apply_gradient),
            "ACCUMULATE_GRADIENT": int(self.accumulate_gradient),
            "WEIGHTS_TRANSPOSED": int(self.weights_transposed),
            "REDUCE_SIZE": self.reduce_size
        })

        self.cl_sources_["all2all/gradient_descent/weights_update"] = {
            "USE_ORTHO": int(bool(self.factor_ortho)),
            "USE_MOMENT": int(bool(self.gradient_moment))
        }

        self.cl_sources_["all2all/gradient_descent/bias_update"] = {
            "BIAS_SIZE": side,
            "OUTPUT_SIZE": batch,
            "USE_MOMENT": int(bool(self.gradient_moment_bias))
        }

        self.build_program(defines, "%s_%d_%d_%d" % (
            self.__class__.__name__, self.input.shape[0],
            self.input.sample_size, self.output.sample_size),
            dtype=dtype)

        self.krn_weights_ = self.get_kernel("weights_update")
        self.krn_weights_.set_args(self.err_output.devmem,
                                   self.input.devmem,
                                   self.weights.devmem,
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
            self.krn_weights_.set_arg(11, self.col_sums.devmem)

        if self.krn_err_output_name:
            self.krn_err_output_ = self.get_kernel(self.krn_err_output_name)
            self.krn_err_output_.set_args(self.err_output.devmem,
                                          self.output.devmem)

    def ocl_init(self):
        dtype = self.err_output.dtype
        block_size = self.device.device_info.get_block_size(
            kernel="matrix_multiplication", dtype=dtype)
        side = self.weights.shape[1 if self.weights_transposed else 0]
        other = self.weights.size // side
        batch = self.input.mem.shape[0]

        if self.need_err_input:
            self.cl_sources_["all2all/gradient_descent/err_input_update"] = {}

        self._gpu_init({"BLOCK_SIZE": block_size})

        if self.krn_err_output_name:
            self._global_size_err_output = [self.err_output.mem.size]
            self._local_size_err_output = None

        if self.need_err_input:
            self.krn_err_input_ = self.get_kernel("err_input_update")
            self.krn_err_input_.set_args(
                self.err_output.devmem, self.weights.devmem,
                self.err_input.devmem)
            self._global_size_err_input = [
                roundup(other, block_size), roundup(batch, block_size)]
            self._local_size_err_input = [block_size, block_size]

        self._global_size_weights = [
            roundup(other, block_size), roundup(side, block_size)]
        self._local_size_weights = [block_size, block_size]

        if self.include_bias:
            self._global_size_bias = [side * self.reduce_size]
            self._local_size_bias = [self.reduce_size]

        self._global_size_ortho = [other * self.reduce_size]
        self._local_size_ortho = [self.reduce_size]

    def cuda_init(self):
        self._gpu_init({})

        side = self.weights.shape[1 if self.weights_transposed else 0]
        other = self.weights.size // side

        if self.krn_err_output_ is not None:
            block_size = self.device.suggest_block_size(self.krn_err_output_)
            self._global_size_err_output = (int(numpy.ceil(
                self.err_output.size / block_size)), 1, 1)
            self._local_size_err_output = (block_size, 1, 1)

        block_size = self.device.suggest_block_size(self.krn_weights_)
        self._global_size_weights = (int(numpy.ceil(
            self.weights.size / block_size)), 1, 1)
        self._local_size_weights = (block_size, 1, 1)

        if self.include_bias:
            self._global_size_bias = (side, 1, 1)
            self._local_size_bias = (self.reduce_size, 1, 1)

        self._global_size_ortho = (other, 1, 1)
        self._local_size_ortho = (self.reduce_size, 1, 1)

        dtype = self.input.dtype
        self.gemm_ = (cublas.CUBLAS.sgemm if dtype == numpy.float32
                      else cublas.CUBLAS.dgemm)
        self.np_one = numpy.ones(1, dtype=dtype)
        self.np_zero = numpy.zeros(1, dtype=dtype)

    def cpu_weights_update(self):
        self.input.map_read()
        self.err_output.map_read()
        self.weights.map_write()
        self.gradient_weights.map_write()
        self.accumulated_gradient_weights.map_write()

        lr = self.learning_rate
        factor_l12 = self.weights_decay
        l1_vs_l2 = self.l1_vs_l2

        err_output = reshape(
            self.err_output.mem,
            [self.err_output.mem.shape[0],
             self.err_output.mem.size // self.err_output.mem.shape[0]])
        inp = reshape(
            self.input.mem, [self.input.mem.shape[0],
                             self.input.mem.size // self.input.mem.shape[0]])
        numpy.dot(err_output.transpose(), inp, self.gradient_weights.mem)

        gradient = -nn_units.GradientDescentBase.cpu_gradient_step(
            self.weights.mem, self.gradient_weights.mem,
            lr, factor_l12, l1_vs_l2, self.factor_ortho)
        if self.accumulate_gradient == self.OP_NONE:
            pass
        elif self.accumulate_gradient == self.OP_STORE:
            self.accumulated_gradient_weights.mem[:] = gradient
        elif self.accumulate_gradient == self.OP_ADD:
            self.accumulated_gradient_weights.mem[:] += gradient
        elif self.accumulate_gradient == self.OP_FLUSH:
            gradient += self.accumulated_gradient_weights.mem
            self.accumulated_gradient_weights.mem[:] = 0
        else:
            raise ValueError("Incorrect accumulate_gradient attribute value")
        if self.gradient_weights_with_moment:
            gradient += (self.gradient_weights_with_moment.mem *
                         self.gradient_moment)
            self.gradient_weights_with_moment.mem[:] = gradient[:]
        if self.apply_gradient:
            if self.weights_transposed:
                self.weights.mem += gradient.transpose()
            else:
                self.weights.mem += gradient

    def cpu_bias_update(self):
        if not self.include_bias:
            return

        self.err_output.map_read()
        self.bias.map_write()
        self.gradient_bias.map_write()
        self.accumulated_gradient_bias.map_write()
        self.gradient_bias_with_moment.map_write()

        lr = self.learning_rate_bias
        factor_l12 = self.weights_decay_bias
        l1_vs_l2 = self.l1_vs_l2_bias

        self.gradient_bias.mem[:] = self.err_output.mem.sum(axis=0)

        gradient = -nn_units.GradientDescentBase.cpu_gradient_step(
            self.bias.mem, self.gradient_bias.mem,
            lr, factor_l12, l1_vs_l2)
        if self.accumulate_gradient == self.OP_NONE:
            pass
        elif self.accumulate_gradient == self.OP_STORE:
            self.accumulated_gradient_bias.mem[:] = gradient
        elif self.accumulate_gradient == self.OP_ADD:
            self.accumulated_gradient_bias.mem[:] += gradient
        elif self.accumulate_gradient == self.OP_FLUSH:
            gradient += self.accumulated_gradient_bias.mem
            self.accumulated_gradient_bias.mem[:] = 0
        else:
            raise ValueError("Incorrect accumulate_gradient attribute value")
        if self.gradient_bias_with_moment:
            gradient += (self.gradient_bias_with_moment.mem *
                         self.gradient_moment)
            self.gradient_bias_with_moment.mem[:] = gradient[:]
        if self.apply_gradient:
            self.bias.mem += gradient

    def cpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return
        self.err_input.map_invalidate()
        self.err_output.map_read()
        self.weights.map_read()
        err_output = reshape(
            self.err_output.mem,
            [self.err_output.mem.shape[0],
             self.err_output.mem.size // self.err_output.mem.shape[0]])
        err_input = reshape(
            self.err_input.mem,
            [self.err_input.mem.shape[0],
             self.err_input.mem.size // self.err_input.mem.shape[0]])
        if self.weights_transposed:
            numpy.dot(err_output, self.weights.mem.transpose(), err_input)
        else:
            numpy.dot(err_output, self.weights.mem, err_input)

    def gpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return
        self.err_input.unmap()
        self.err_output.unmap()
        self.weights.unmap()
        self.execute_kernel(
            self._global_size_err_input, self._local_size_err_input,
            self.krn_err_input_)

    def cpu_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.cpu_err_output_update()
        self.cpu_err_input_update()
        self.cpu_weights_update()
        self.cpu_bias_update()
        self.print_debug_data(t1)

    def ocl_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_output_update()
        if self.prefer_numpy:
            self.cpu_err_input_update()
        else:
            self.gpu_err_input_update()
        self.gpu_weights_update()
        self.gpu_bias_update()
        self.print_debug_data(t1)

    def cuda_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_output_update()
        self.cuda_err_input_update()
        self.cuda_weights_update()
        self.gpu_bias_update()
        self.print_debug_data(t1)

    def cuda_err_input_update(self):
        if not self.need_err_input:
            return

        self.err_output.unmap()
        self.weights.unmap()
        self.err_input.unmap()

        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_T
            if self.weights_transposed else cublas.CUBLAS_OP_N,
            cublas.CUBLAS_OP_N,
            self.err_input.sample_size, self.err_output.shape[0],
            self.err_output.sample_size,
            self.np_one, self.weights.devmem, self.err_output.devmem,
            self.np_zero, self.err_input.devmem)

    def cuda_weights_update(self):
        self.err_output.unmap()
        self.gradient_weights.unmap()
        self.input.unmap()

        if self.weights_transposed:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.err_output.sample_size, self.input.sample_size,
                self.err_output.shape[0],
                self.np_one, self.err_output.devmem, self.input.devmem,
                self.np_zero, self.gradient_weights.devmem)
        else:
            self.gemm_(
                self.device.blas, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                self.input.sample_size, self.err_output.sample_size,
                self.err_output.shape[0],
                self.np_one, self.input.devmem, self.err_output.devmem,
                self.np_zero, self.gradient_weights.devmem)

        # Accumulate/apply gradient
        self.gpu_weights_update()


class GDSM(GradientDescent):
    """Gradient Descent for :class:`veles.znicz.all2all.All2AllSoftmax`.

    We minimize cross-entropy error function for softmax, so gradient descent
    is the same as in :class:`veles.znicz.gd.GradientDescent`.
    """
    MAPPING = {"softmax"}


class GDTanh(GradientDescent):
    """Gradient Descent for

    :math:`f(x) = 1.7159 \\tanh(0.6666  s), s = (W  x + b)`,
        :math:`y = a \cdot \\tanh(b s)`

    :math:`f'(s) = (a \\cdot \\tanh(b  s))' = a \\cdot \\tanh'(b  s) \\cdot b`

    :math:`= a (1 - \\tanh^2(b s)) * b  =  a b - a * b * \\tanh^2(b s)`

    :math:`= a b - y * y * b / a =  y^2 (-b / a) + (a \\cdot b)`

    :math:`z = y^2 (-0.388484177) + 1.14381894`
    """

    MAPPING = {"all2all_tanh"}

    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative
        by s in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= output * output * (-0.388484177) + 1.14381894

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_tanh"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        super(GDTanh, self).initialize(device=device, **kwargs)


class GDRELU(GradientDescent):
    """
    Gradient Descent for :math:`f(x) = \\log(1 + \\exp(s))`

    :math:`s = (W x + b)`

    :math:`y = \\log(1.0 + \\exp(s))`

    :math:`f'(s) = \\frac{1}{1 + \\exp(-s)} = 1 - \\exp(-y)`
    """

    MAPPING = {"all2all_relu"}

    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= 1.0 - numpy.exp(-output)

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_relu"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        super(GDRELU, self).initialize(device=device, **kwargs)


class GDStrictRELU(GradientDescent):
    """Gradient Descent for strict ReLU (like in CAFFE)

    :math:`f(x) = \\max(x, 0)`

    :math:`f'(s) = \\begin{cases}1 & s > 0 \\\\ 0 & else. \\\\ \\end{cases}`
    """

    MAPPING = {"all2all_str"}

    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= numpy.greater(output, 0)

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_strict_relu"] = {
            "ERR_OUTPUT_SIZE": self.err_output.size}
        self.krn_err_output_name = "err_y_update"
        super(GDRELU, self).initialize(device=device, **kwargs)
