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

import numpy
import time
from zope.interface import implementer

import veles.error as error
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

    def initialize(self, device, **kwargs):
        super(GradientDescent, self).initialize(device=device, **kwargs)

        if self.err_output.shape != self.output.shape:
            raise error.BadFormatError("err_output.shape != output.shape")

        if (self.need_err_input and
            (self.err_input.mem is None or
             self.err_input.mem.size != self.input.mem.size)):
            self.err_input.reset()
            self.err_input.mem = numpy.zeros(self.input.mem.shape,
                                             dtype=self.err_output.mem.dtype)

        if (self.store_gradient and
            (self.gradient_weights.mem is None or
             self.gradient_weights.mem.size != self.weights.mem.size)):
            self.gradient_weights.reset()
            self.gradient_weights.mem = numpy.zeros_like(self.weights.mem)

        if (self.include_bias and self.store_gradient and
            (self.gradient_bias.mem is None or
             self.gradient_bias.mem.size != self.bias.mem.size)):
            self.gradient_bias.reset()
            self.gradient_bias.mem = numpy.zeros_like(self.bias.mem)

        self.weights.initialize(self.device)
        self.bias.initialize(self.device)
        self.output.initialize(self.device)
        self.input.initialize(self.device)
        self.err_output.initialize(self.device)
        self.err_input.initialize(self.device)
        if self.store_gradient:
            self.gradient_weights.initialize(self.device)
            self.gradient_bias.initialize(self.device)

        self.backend_init()

    def ocl_init(self):
        dtype = self.err_output.mem.dtype
        self.cl_const = numpy.zeros(5, dtype=dtype)

        side = self.weights.shape[1 if self.weights_transposed else 0]
        other = self.weights.size // side
        assert other == self.input.sample_size
        assert side == self.output.sample_size
        if self.factor_ortho:
            if not self.col_sums or self.col_sums.size < other:
                self.col_sums.reset()
                self.col_sums.mem = numpy.zeros(other, dtype=dtype)
            self.col_sums.initialize(self.device)
        self.reduce_size = roundup(min(self.reduce_size, other), 32)

        batch = self.input.mem.shape[0]
        block_size = self.device.device_info.get_block_size(
            kernel="matrix_multiplication", dtype=self.err_output.dtype)
        defines = {
            "H": other,
            "Y": side,
            "BATCH": batch,
            "APPLY_GRADIENT": int(self.apply_gradient),
            "STORE_GRADIENT": int(self.store_gradient),
            "WEIGHTS_TRANSPOSED": int(self.weights_transposed),
            "REDUCE_SIZE": self.reduce_size,
            "BLOCK_SIZE": block_size,
            "USE_ORTHO": int(bool(self.factor_ortho))
        }

        self.cl_sources_["all2all/gradient_descent/err_input_update"] = {}
        self._global_size_err_input = [
            roundup(other, block_size), roundup(batch, block_size)]
        self._local_size_err_input = [block_size, block_size]

        self.cl_sources_["all2all/gradient_descent/weights_update"] = {}
        self._global_size_weights = [
            roundup(other, block_size), roundup(side, block_size)]
        self._local_size_weights = [block_size, block_size]

        self.cl_sources_["all2all/gradient_descent/bias_update"] = {}
        self._global_size_bias = [side * self.reduce_size]
        self._local_size_bias = [self.reduce_size]

        self.build_program(defines, "%s_%d_%d_%d" % (
            self.__class__.__name__, self.input.shape[0],
            self.input.sample_size, self.output.sample_size),
            dtype=dtype)

        if self.need_err_input:
            self.krn_err_input_ = self.get_kernel("err_input_update")
            self.krn_err_input_.set_args(
                self.err_output.devmem, self.weights.devmem,
                self.err_input.devmem)

        self.krn_weights_ = self.get_kernel("weights_update")
        self.krn_weights_.set_args(self.err_output.devmem,
                                   self.input.devmem,
                                   self.weights.devmem,
                                   self.gradient_weights.devmem)

        if self.include_bias:
            self.krn_bias_ = self.get_kernel("bias_update")
            self.krn_bias_.set_args(
                self.err_output.devmem, self.bias.devmem,
                self.gradient_bias.devmem)

        if self.factor_ortho:
            self.krn_compute_col_sums_ = self.get_kernel("compute_col_sums")
            self.krn_compute_col_sums_.set_args(self.weights.devmem,
                                                self.col_sums.devmem)
            self.krn_weights_.set_arg(9, self.col_sums.devmem)

    def cpu_weights_update(self):
        self.input.map_read()
        self.err_output.map_read()
        self.weights.map_write()
        self.gradient_weights.map_write()

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
        gradient = -nn_units.GradientDescentBase.cpu_gradient_step(
            self.weights.mem, numpy.dot(err_output.transpose(), inp),
            lr, factor_l12, l1_vs_l2, self.factor_ortho)
        if self.store_gradient:
            gradient += self.gradient_weights.mem * self.gradient_moment
            self.gradient_weights.mem[:] = gradient[:]
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

        lr = self.learning_rate_bias
        factor_l12 = self.weights_decay_bias
        l1_vs_l2 = self.l1_vs_l2_bias

        gradient = -nn_units.GradientDescentBase.cpu_gradient_step(
            self.bias.mem, self.err_output.mem.sum(axis=0),
            lr, factor_l12, l1_vs_l2)
        if self.store_gradient:
            gradient += self.gradient_bias.mem * self.gradient_moment
            self.gradient_bias.mem[:] = gradient[:]
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

    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative by output.
        """
        pass

    def gpu_err_output_update(self):
        """Multiply err_output by activation derivative by output.
        """
        if self.krn_err_output_ is None:
            return
        self.output.unmap()
        self.err_output.unmap()
        self.execute_kernel([self.err_output.mem.size], None,
                            self.krn_err_output_)

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
        self.cl_sources_["gradient_descent_tanh"] = {}
        super(GDTanh, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_args(self.err_output.devmem,
                                      self.output.devmem)


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
        self.cl_sources_["gradient_descent_relu"] = {}
        super(GDRELU, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_args(self.err_output.devmem,
                                      self.output.devmem)
