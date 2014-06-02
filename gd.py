"""
Created on Apr 15, 2013

Gradient Descent Units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import numpy
import logging
import opencl4py as cl
import time
from zope.interface import implementer

from veles.config import root
from veles.external.prettytable import PrettyTable
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.opencl_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
class GradientDescent(nn_units.GradientDescentBase):
    """Gradient Descent.

    Must be assigned before initialize():
        output
        input
        err_output
        weights
        bias
        batch_size

    Updates after run():
        err_input
        err_outpur
        weights
        bias

    Creates within initialize():
        err_input

    Attributes:
        krn_err_input_: OpenCL kernel for matrix multiplication.
        krn_weights_: OpenCL kernel for weights update.
        krn_err_output_: OpenCL kernel for err_output update.
        krn_bias_: OpenCL kernel for bias update.
    """
    def __init__(self, workflow, **kwargs):
        super(GradientDescent, self).__init__(workflow, **kwargs)
        self.cl_const = numpy.zeros(3, dtype=opencl_types.dtypes[
            root.common.dtype])
        self.reduce_size = 64

    def init_unpickled(self):
        super(GradientDescent, self).init_unpickled()
        self.cl_sources_["gradient_descent.cl"] = {}
        self.krn_err_input_ = None
        self.krn_weights_ = None
        self.krn_err_output_ = None
        self.krn_bias_ = None

    def initialize(self, device, **kwargs):
        super(GradientDescent, self).initialize(device=device, **kwargs)
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

        if (self.store_gradient and
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

        if self.device is None:
            return

        if self.program_ is None:
            block_size = self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.err_output.mem.dtype)]
            self.reduce_size = min(self.reduce_size, self.bias.mem.size)

            defines = {
                'BLOCK_SIZE': block_size,
                'BATCH': self.input.mem.shape[0],
                'H': self.input.mem.size // self.input.mem.shape[0],
                'Y': self.output.mem.size // self.output.mem.shape[0],
                'REDUCE_SIZE': self.reduce_size
            }
            if self.apply_gradient:
                defines['APPLY_GRADIENT'] = 1
            if self.weights_transposed:
                defines['WEIGHTS_TRANSPOSED'] = 1
            if self.store_gradient:
                defines['STORE_GRADIENT'] = 1

            self.build_program(defines, "gd_%d_%d.cl" % (
                self.input.mem.size // self.input.mem.shape[0],
                self.output.mem.size // self.output.mem.shape[0]),
                dtype=self.err_output.mem.dtype)

            if self.need_err_input:
                self.krn_err_input_ = self.get_kernel("err_h_update")
                self.krn_err_input_.set_args(
                    self.err_output.devmem, self.weights.devmem,
                    self.err_input.devmem)

            self.krn_weights_ = self.get_kernel("weights_update")
            self.krn_weights_.set_args(self.err_output.devmem,
                                       self.input.devmem,
                                       self.weights.devmem,
                                       self.gradient_weights.devmem)

            self.krn_bias_ = self.get_kernel("bias_update")
            self.krn_bias_.set_args(self.err_output.devmem, self.bias.devmem,
                                    self.gradient_bias.devmem)

    def cpu_weights_update(self):
        self.input.map_read()
        self.err_output.map_read()
        self.weights.map_write()
        self.bias.map_write()
        self.gradient_weights.map_invalidate()
        self.gradient_bias.map_invalidate()

        if self.batch_size is None:
            batch_size = self.output.mem.shape[0]
        else:
            batch_size = int(self.batch_size)

        # weights
        alpha_batch = -self.learning_rate / batch_size
        alpha_lambda = -self.learning_rate * self.weights_decay

        err_output = formats.reshape(
            self.err_output.mem,
            [self.err_output.mem.shape[0],
             self.err_output.mem.size // self.err_output.mem.shape[0]])
        inp = formats.reshape(
            self.input.mem, [self.input.mem.shape[0],
                             self.input.mem.size // self.input.mem.shape[0]])
        gradient = numpy.dot(err_output.transpose(), inp)
        gradient *= alpha_batch
        gradient += self.weights.mem * alpha_lambda
        if self.store_gradient:
            gradient += self.gradient_weights.mem * self.gradient_moment
            self.gradient_weights.mem[:] = gradient[:]
        if self.apply_gradient:
            if self.weights_transposed:
                self.weights.mem += gradient.transpose()
            else:
                self.weights.mem += gradient

        # bias
        alpha_batch = -self.learning_rate_bias / batch_size
        alpha_lambda = -self.learning_rate_bias * self.weights_decay_bias

        gradient = err_output.sum(axis=0) * alpha_batch
        gradient += self.bias.mem * alpha_lambda
        if self.store_gradient:
            gradient += self.gradient_bias.mem * self.gradient_moment
            self.gradient_bias.mem[:] = gradient[:]
        if self.apply_gradient:
            self.bias.mem += gradient

    def gpu_weights_update(self):
        self.input.unmap()
        self.err_output.unmap()
        self.weights.unmap()
        self.bias.unmap()
        self.gradient_weights.unmap()
        self.gradient_bias.unmap()

        if self.batch_size is None:
            batch_size = self.output.mem.shape[0]
        else:
            batch_size = int(self.batch_size)
        self.cl_const[0] = -self.learning_rate / batch_size
        self.cl_const[1] = -self.learning_rate * self.weights_decay
        self.cl_const[2] = self.gradient_moment
        self.krn_weights_.set_args(cl.skip(4), self.cl_const[0:1],
                                   self.cl_const[1:2], self.cl_const[2:3])
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_output.mem.dtype)]
        if self.weights_transposed:
            global_size = [
                formats.roundup(
                    self.err_output.mem.size // self.err_output.mem.shape[0],
                    block_size),
                formats.roundup(self.input.mem.size // self.input.mem.shape[0],
                                block_size)]
        else:
            global_size = [
                formats.roundup(self.input.mem.size // self.input.mem.shape[0],
                                block_size),
                formats.roundup(
                    self.err_output.mem.size // self.err_output.mem.shape[0],
                    block_size)]
        local_size = [block_size, block_size]
        ev1 = self.execute_kernel(global_size, local_size, self.krn_weights_)

        self.cl_const[0] = -self.learning_rate_bias / batch_size
        self.cl_const[1] = -self.learning_rate_bias * self.weights_decay_bias
        self.cl_const[2] = self.gradient_moment_bias
        self.krn_bias_.set_args(cl.skip(3), self.cl_const[0:1],
                                self.cl_const[1:2], self.cl_const[2:3])
        global_size = [(self.err_output.mem.size //
                        self.err_output.mem.shape[0]) *
                       self.reduce_size]
        local_size = [self.reduce_size]
        ev2 = self.execute_kernel(global_size, local_size, self.krn_bias_)

        ev1.wait()
        ev2.wait()

    def cpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return
        self.err_input.map_invalidate()
        self.err_output.map_read()
        self.weights.map_read()
        err_output = formats.reshape(
            self.err_output.mem,
            [self.err_output.mem.shape[0],
             self.err_output.mem.size // self.err_output.mem.shape[0]])
        err_input = formats.reshape(
            self.err_input.mem,
            [self.err_input.mem.shape[0],
             self.err_input.mem.size // self.err_input.mem.shape[0]])
        if self.weights_transposed:
            err_input[:] = numpy.dot(err_output,
                                     self.weights.mem.transpose())[:]
        else:
            err_input[:] = numpy.dot(err_output, self.weights.mem)[:]

    def gpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return
        self.err_input.unmap()
        self.err_output.unmap()
        self.weights.unmap()
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_output.mem.dtype)]
        global_size = [formats.roundup(self.err_input.mem.size //
                       self.err_input.mem.shape[0], block_size),
                       formats.roundup(self.err_input.mem.shape[0],
                                       block_size)]
        local_size = [block_size, block_size]
        event = self.execute_kernel(global_size, local_size,
                                    self.krn_err_input_)
        event.wait()

    def print_debug_data(self, t_start):
        """
        Show weights statistics
        """
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        self.weights.map_read()
        self.bias.map_read()
        self.gradient_bias.map_read()
        self.gradient_weights.map_read()
        weights = self.weights.mem
        bias = self.bias.mem
        grad_weights = self.gradient_weights.mem
        grad_bias = self.gradient_bias.mem

        n_input = self.input.mem.size // self.input.mem.shape[0]
        n_output = self.output.mem.size // self.output.mem.shape[0]
        delta_time = time.time() - t_start

        stats_table = PrettyTable("n_input", "n_output", "time")
        stats_table.float_format = ".3"
        stats_table.add_row(n_input, n_output, delta_time)
        self.debug("\n" + stats_table.get_string())

        weight_table = PrettyTable("TYPE", "Mean", "StdDev", "Min", "Max")
        weight_table.float_format = ".10"
        for (w_name, w_array) in [("Weight", weights), ("Bias", bias),
                                  ("Grad Weight", grad_weights),
                                  ("Grad Bias", grad_bias)]:
            w_mean = w_stddev = w_min = w_max = None
            if w_array is not None and w_array.size > 0:
                w_mean = numpy.mean(w_array)
                w_stddev = numpy.std(w_array)
                w_min = numpy.min(w_array)
                w_max = numpy.max(w_array)
            weight_table.add_row(w_name, w_mean, w_stddev, w_min, w_max)
        self.debug("\n" + weight_table.get_string())

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
        ev = self.execute_kernel([self.err_output.mem.size], None,
                                 self.krn_err_output_)
        ev.wait()

    def cpu_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.cpu_err_output_update()
        self.cpu_err_input_update()
        self.cpu_weights_update()
        self.print_debug_data(t1)

    def ocl_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_output_update()
        self.gpu_err_input_update()
        self.gpu_weights_update()
        self.print_debug_data(t1)


class GDSM(GradientDescent):
    """Gradient Descent for SoftMax.

    We minimize cross-entropy error function for softmax,
    so gradient descent is the same as in GradientDescent.
    """
    pass


class GDTanh(GradientDescent):
    """Gradient Descent for f(x) = 1.7159 * tanh(0.6666 * s), s = (W * x + b),
       y = a * tanh(b * s).

    f'(s) = (a * tanh(b * s))' = a * tanh'(b * s) * b
          = a * (1.0 - tanh^2(b * s)) * b
          = a * b - a * b * tanh^2(b * s)
          = a * b - y * y * b / a
          = y * y * (-b / a) + (a * b)
          = y * y * (-0.388484177) + 1.14381894
    """
    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative
        by s in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= output * output * (-0.388484177) + 1.14381894

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_tanh.cl"] = {}
        super(GDTanh, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_args(self.err_output.devmem,
                                      self.output.devmem)


class GDRELU(GradientDescent):
    """Gradient Descent for f(x) = log(1.0 + exp(s)), s = (W * x + b),
       y = log(1.0 + exp(s)).

    f'(s) = 1.0 / (1.0 + exp(-s)) = 1.0 - exp(-y)
    """
    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= 1.0 - numpy.exp(-output)

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_relu.cl"] = {}
        super(GDRELU, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_args(self.err_output.devmem,
                                      self.output.devmem)
