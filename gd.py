"""
Created on Apr 15, 2013

Gradient Descent Units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import logging
import time

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units


class GradientDescent(nn_units.GradientDescentBase):
    """Gradient Descent.

    Should be assigned before initialize():
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
        if (self.err_input.v is None or
                self.err_input.v.size != self.input.v.size):
            self.err_input.reset()
            self.err_input.v = numpy.zeros(self.input.v.shape,
                                           dtype=self.err_output.v.dtype)

        if (self.store_gradient and
            (self.gradient_weights.v is None or
             self.gradient_weights.v.size != self.weights.v.size)):
            self.gradient_weights.reset()
            self.gradient_weights.v = numpy.zeros_like(self.weights.v)

        if (self.store_gradient and
            (self.gradient_bias.v is None or
             self.gradient_bias.v.size != self.bias.v.size)):
            self.gradient_bias.reset()
            self.gradient_bias.v = numpy.zeros_like(self.bias.v)

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
                opencl_types.numpy_dtype_to_opencl(self.err_output.v.dtype)]
            self.reduce_size = min(self.reduce_size, self.bias.v.size)

            defines = {
                'BLOCK_SIZE': block_size,
                'BATCH': self.err_input.v.shape[0],
                'H': self.err_input.v.size // self.err_input.v.shape[0],
                'Y': self.err_output.v.size // self.err_output.v.shape[0],
                'REDUCE_SIZE': self.reduce_size
            }
            if self.apply_gradient:
                defines['APPLY_GRADIENT'] = 1
            if self.weights_transposed:
                defines['WEIGHTS_TRANSPOSED'] = 1
            if self.store_gradient:
                defines['STORE_GRADIENT'] = 1

            self.build_program(defines, "gd_%d_%d.cl" % (
                self.input.v.size // self.input.v.shape[0],
                self.output.v.size // self.output.v.shape[0]),
                dtype=self.err_output.v.dtype)

            self.krn_err_input_ = self.get_kernel("err_h_update")
            self.krn_err_input_.set_arg(0, self.err_output.v_)
            self.krn_err_input_.set_arg(1, self.weights.v_)
            self.krn_err_input_.set_arg(2, self.err_input.v_)

            self.krn_weights_ = self.get_kernel("weights_update")
            self.krn_weights_.set_arg(0, self.err_output.v_)
            self.krn_weights_.set_arg(1, self.input.v_)
            self.krn_weights_.set_arg(2, self.weights.v_)
            self.krn_weights_.set_arg(3, self.gradient_weights.v_)

            self.krn_bias_ = self.get_kernel("bias_update")
            self.krn_bias_.set_arg(0, self.err_output.v_)
            self.krn_bias_.set_arg(1, self.bias.v_)
            self.krn_bias_.set_arg(2, self.gradient_bias.v_)

    def cpu_weights_update(self):
        self.input.map_read()
        self.err_output.map_read()
        self.weights.map_write()
        self.bias.map_write()
        self.gradient_weights.map_invalidate()
        self.gradient_bias.map_invalidate()

        batch_size = self.batch_size or self.output.v.shape[0]

        alpha_batch = -self.learning_rate / batch_size
        alpha_lambda = -self.learning_rate * self.weights_decay

        err_output = formats.reshape(
            self.err_output.v,
            [self.err_output.v.shape[0],
             self.err_output.v.size // self.err_output.v.shape[0]])
        input = formats.reshape(
            self.input.v, [self.input.v.shape[0],
                       self.input.v.size // self.input.v.shape[0]])
        gradient = numpy.dot(err_output.transpose(), input)
        gradient *= alpha_batch
        gradient += self.weights.v * alpha_lambda
        if self.store_gradient:
            gradient += self.gradient_weights.v * self.gradient_moment
            self.gradient_weights.v[:] = gradient[:]
        if self.weights_transposed:
            self.weights.v += gradient.transpose()
        else:
            self.weights.v += gradient

        gradient = err_output.sum(axis=0) * alpha_batch
        if self.store_gradient:
            gradient += self.gradient_bias.v * self.gradient_moment
            self.gradient_bias.v[:] = gradient[:]
        self.bias.v += gradient

    def gpu_weights_update(self):
        self.input.unmap()
        self.err_output.unmap()
        self.weights.unmap()
        self.bias.unmap()
        self.gradient_weights.unmap()
        self.gradient_bias.unmap()

        batch_size = self.batch_size or self.output.v.shape[0]
        self.cl_const[0] = -self.learning_rate / batch_size
        self.cl_const[1] = -self.learning_rate * self.weights_decay
        self.cl_const[2] = self.gradient_moment
        self.krn_weights_.set_arg(4, self.cl_const[0:1])
        self.krn_weights_.set_arg(5, self.cl_const[1:2])
        self.krn_weights_.set_arg(6, self.cl_const[2:3])
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_output.v.dtype)]
        if self.weights_transposed:
            global_size = [
                formats.roundup(
                    self.err_output.v.size // self.err_output.v.shape[0],
                    block_size),
                formats.roundup(self.input.v.size // self.input.v.shape[0],
                                block_size)]
        else:
            global_size = [
                formats.roundup(self.input.v.size // self.input.v.shape[0],
                                block_size),
                formats.roundup(
                    self.err_output.v.size // self.err_output.v.shape[0],
                    block_size)]
        local_size = [block_size, block_size]
        ev1 = self.execute_kernel(self.krn_weights_, global_size, local_size)

        self.krn_bias_.set_arg(3, self.cl_const[0:1])
        self.krn_bias_.set_arg(4, self.cl_const[2:3])
        global_size = [(self.err_output.v.size // self.err_output.v.shape[0]) *
                       self.reduce_size]
        local_size = [self.reduce_size]
        ev2 = self.execute_kernel(self.krn_bias_, global_size, local_size)

        ev1.wait()
        ev2.wait()

    def cpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        self.err_input.map_invalidate()
        self.err_output.map_read()
        self.weights.map_read()
        err_output = formats.reshape(
            self.err_output.v,
            [self.err_output.v.shape[0],
             self.err_output.v.size // self.err_output.v.shape[0]])
        err_input = formats.reshape(
            self.err_input.v, [self.err_input.v.shape[0],
                           self.err_input.v.size // self.err_input.v.shape[0]])
        if self.weights_transposed:
            err_input[:] = numpy.dot(err_output, self.weights.v.transpose())[:]
        else:
            err_input[:] = numpy.dot(err_output, self.weights.v)[:]

    def gpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        self.err_input.unmap()
        self.err_output.unmap()
        self.weights.unmap()
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_output.v.dtype)]
        global_size = [formats.roundup(self.err_input.v.size //
                       self.err_input.v.shape[0], block_size),
                       formats.roundup(self.err_input.v.shape[0],
                                       block_size)]
        local_size = [block_size, block_size]
        event = self.execute_kernel(self.krn_err_input_, global_size,
                                    local_size)
        event.wait()

    def print_times(self, t_start):
        if not self.log.isEnabledFor(logging.DEBUG):
            return
        self.weights.map_read()
        self.bias.map_read()
        weights = self.weights.v
        bias = self.bias.v
        if weights.dtype in (numpy.complex64, numpy.complex128):
            self.debug(
                "BP %d_%d in %.2f sec: min avg max: "
                "W: %.6f %.6f %.6f B: %.6f %.6f %.6f" %
                (self.input.v.size // self.input.v.shape[0],
                 self.output.v.size // self.output.v.shape[0],
                 time.time() - t_start,
                 min(weights.real.min(), weights.imag.min()),
                 (numpy.average(weights.real) +
                  numpy.average(weights.imag)) * 0.5,
                 max(weights.real.max(), weights.imag.max()),
                 min(bias.real.min(), bias.imag.min()),
                 (numpy.average(bias.real) + numpy.average(bias.imag)) * 0.5,
                 max(bias.real.max(), bias.imag.max())))
        else:
            self.debug(
                "BP %d_%d in %.2f sec: min avg max: "
                "W: %.6f %.6f %.6f B: %.6f %.6f %.6f" %
                (self.input.v.size // self.input.v.shape[0],
                 self.output.v.size // self.output.v.shape[0],
                 time.time() - t_start,
                 weights.min(),
                 numpy.average(weights),
                 weights.max(),
                 bias.min(),
                 numpy.average(bias),
                 bias.max()))

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
        ev = self.execute_kernel(self.krn_err_output_,
                                 [self.err_output.v.size], None)
        ev.wait()

    def cpu_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.cpu_err_output_update()
        self.cpu_err_input_update()
        self.cpu_weights_update()
        self.print_times(t1)

    def ocl_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_output_update()
        self.gpu_err_input_update()
        self.gpu_weights_update()
        self.print_times(t1)


class GDSM(GradientDescent):
    """Gradient Descent for softmax.

    It is the same as GradientDescent inside.
    """


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
        output = self.output.v
        self.err_output.v *= output * output * (-0.388484177) + 1.14381894

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_tanh.cl"] = {}
        super(GDTanh, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_arg(0, self.err_output.v_)
        self.krn_err_output_.set_arg(1, self.output.v_)


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
        output = self.output.v
        self.err_output.v *= 1.0 - numpy.exp(-output)

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_relu.cl"] = {}
        super(GDRELU, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_arg(0, self.err_output.v_)
        self.krn_err_output_.set_arg(1, self.output.v_)
