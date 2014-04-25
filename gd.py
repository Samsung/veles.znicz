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


class GD(nn_units.GD):
    """Gradient Descent.

    Should be assigned before initialize():
        y
        h
        err_y
        weights
        bias
        batch_size

    Updates after run():
        err_h
        err_y
        weights
        bias

    Creates within initialize():
        err_h

    Attributes:
        weights: weights of the current layer.
        bias: bias of the current layer.
        y: output of the current layer as batch of 1D samples.
        h: input of the current layer as batch of 1D samples.
        err_y: backpropagation errors for y.
        err_h: backpropagation errors for h (will compute its).
        krn_err_h_: OpenCL kernel for matrix multiplication.
        krn_weights_: OpenCL kernel for weights update.
        krn_err_y_: OpenCL kernel for err_y update.
        krn_bias_: OpenCL kernel for bias update.
    """
    def __init__(self, workflow, **kwargs):
        super(GD, self).__init__(workflow, **kwargs)
        self.weights = None  # formats.Vector()
        self.bias = None  # formats.Vector()
        self.y = None  # formats.Vector()
        self.h = None  # formats.Vector()
        self.err_y = None  # formats.Vector()
        self.err_h = formats.Vector()
        self.cl_const = numpy.zeros(2, dtype=opencl_types.dtypes[
            root.common.dtype])
        self.reduce_size = 64

    def init_unpickled(self):
        super(GD, self).init_unpickled()
        self.cl_sources_["gradient_descent.cl"] = {}
        self.krn_err_h_ = None
        self.krn_weights_ = None
        self.krn_err_y_ = None
        self.krn_bias_ = None

    def initialize(self):
        super(GD, self).initialize()
        if (self.err_h.v is None or self.err_h.v.size != self.h.v.size):
            self.err_h.reset()
            self.err_h.v = numpy.zeros(self.h.v.shape,
                                       dtype=self.err_y.v.dtype)

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
        self.y.initialize(self.device)
        self.h.initialize(self.device)
        self.err_y.initialize(self.device)
        self.err_h.initialize(self.device)
        if self.store_gradient:
            self.gradient_weights.initialize(self.device)
            self.gradient_bias.initialize(self.device)

        if self.device is None:
            return

        if self.program_ is None:
            block_size = self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.err_y.v.dtype)]
            self.reduce_size = min(self.reduce_size, self.bias.v.size)

            defines = {
                'BLOCK_SIZE': block_size,
                'BATCH': self.err_h.v.shape[0],
                'H': self.err_h.v.size // self.err_h.v.shape[0],
                'Y': self.err_y.v.size // self.err_y.v.shape[0],
                'REDUCE_SIZE': self.reduce_size
            }
            if self.apply_gradient:
                defines['APPLY_GRADIENT'] = 1
            if self.weights_transposed:
                defines['WEIGHTS_TRANSPOSED'] = 1
            if self.store_gradient:
                defines['STORE_GRADIENT'] = 1
            self.build_program(defines, "%s/gd_%d_%d.cl" % (
                root.common.cache_dir,
                self.h.v.size // self.h.v.shape[0],
                self.y.v.size // self.y.v.shape[0]),
                dtype=self.err_y.v.dtype)

            self.krn_err_h_ = self.get_kernel("err_h_update")
            self.krn_err_h_.set_arg(0, self.err_y.v_)
            self.krn_err_h_.set_arg(1, self.weights.v_)
            self.krn_err_h_.set_arg(2, self.err_h.v_)

            self.krn_weights_ = self.get_kernel("weights_update")
            self.krn_weights_.set_arg(0, self.err_y.v_)
            self.krn_weights_.set_arg(1, self.h.v_)
            self.krn_weights_.set_arg(2, self.weights.v_)
            self.krn_weights_.set_arg(3, self.gradient_weights.v_)

            self.krn_bias_ = self.get_kernel("bias_update")
            self.krn_bias_.set_arg(0, self.err_y.v_)
            self.krn_bias_.set_arg(1, self.bias.v_)
            self.krn_bias_.set_arg(2, self.gradient_bias.v_)

    def cpu_weights_update(self):
        self.h.map_read()
        self.err_y.map_read()
        self.weights.map_write()
        self.bias.map_write()
        self.gradient_weights.map_invalidate()
        self.gradient_bias.map_invalidate()

        batch_size = self.batch_size or self.y.v.shape[0]

        alpha_batch = -self.global_alpha / batch_size
        alpha_lambda = -self.global_alpha * self.global_lambda

        err_y = formats.reshape(
            self.err_y.v, [self.err_y.v.shape[0],
                           self.err_y.v.size // self.err_y.v.shape[0]])
        h = formats.reshape(
            self.h.v, [self.h.v.shape[0],
                       self.h.v.size // self.h.v.shape[0]])
        gradient = numpy.dot(err_y.transpose(), h)
        gradient *= alpha_batch
        gradient += self.weights.v * alpha_lambda
        if self.store_gradient:
            self.gradient_weights.v += gradient
        if self.weights_transposed:
            self.weights.v += gradient.transpose()
        else:
            self.weights.v += gradient

        gradient = err_y.sum(axis=0) * alpha_batch
        if self.store_gradient:
            self.gradient_bias.v += gradient
        self.bias.v += gradient

    def gpu_weights_update(self):
        self.h.unmap()
        self.err_y.unmap()
        self.weights.unmap()
        self.bias.unmap()
        self.gradient_weights.unmap()
        self.gradient_bias.unmap()

        batch_size = self.batch_size or self.y.v.shape[0]
        self.cl_const[0] = -self.global_alpha / batch_size
        self.cl_const[1] = -self.global_alpha * self.global_lambda
        self.krn_weights_.set_arg(4, self.cl_const[0:1])
        self.krn_weights_.set_arg(5, self.cl_const[1:2])
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_y.v.dtype)]
        if self.weights_transposed:
            global_size = [
                formats.roundup(self.err_y.v.size // self.err_y.v.shape[0],
                                block_size),
                formats.roundup(self.h.v.size // self.h.v.shape[0],
                                block_size)]
        else:
            global_size = [
                formats.roundup(self.h.v.size // self.h.v.shape[0],
                                block_size),
                formats.roundup(self.err_y.v.size // self.err_y.v.shape[0],
                                block_size)]
        local_size = [block_size, block_size]
        ev1 = self.execute_kernel(self.krn_weights_, global_size, local_size)

        self.krn_bias_.set_arg(3, self.cl_const[0:1])
        global_size = [(self.err_y.v.size // self.err_y.v.shape[0]) *
                       self.reduce_size]
        local_size = [self.reduce_size]
        ev2 = self.execute_kernel(self.krn_bias_, global_size, local_size)

        ev1.wait()
        ev2.wait()

    def cpu_err_h_update(self):
        """Backpropagate error (will compute err_h).
        """
        self.err_h.map_invalidate()
        self.err_y.map_read()
        self.weights.map_read()
        err_y = formats.reshape(
            self.err_y.v, [self.err_y.v.shape[0],
                           self.err_y.v.size // self.err_y.v.shape[0]])
        err_h = formats.reshape(
            self.err_h.v, [self.err_h.v.shape[0],
                           self.err_h.v.size // self.err_h.v.shape[0]])
        if self.weights_transposed:
            err_h[:] = numpy.dot(err_y, self.weights.v.transpose())[:]
        else:
            err_h[:] = numpy.dot(err_y, self.weights.v)[:]

    def gpu_err_h_update(self):
        """Backpropagate error (will compute err_h).
        """
        self.err_h.unmap()
        self.err_y.unmap()
        self.weights.unmap()
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_y.v.dtype)]
        global_size = [formats.roundup(self.err_h.v.size //
                       self.err_h.v.shape[0], block_size),
                       formats.roundup(self.err_h.v.shape[0],
                                       block_size)]
        local_size = [block_size, block_size]
        event = self.execute_kernel(self.krn_err_h_, global_size, local_size)
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
                (self.h.v.size // self.h.v.shape[0],
                 self.y.v.size // self.y.v.shape[0],
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
                (self.h.v.size // self.h.v.shape[0],
                 self.y.v.size // self.y.v.shape[0],
                 time.time() - t_start,
                 weights.min(),
                 numpy.average(weights),
                 weights.max(),
                 bias.min(),
                 numpy.average(bias),
                 bias.max()))

    def cpu_err_y_update(self):
        """Multiply err_y by activation derivative by y.
        """
        pass

    def gpu_err_y_update(self):
        """Multiply err_y by activation derivative by y.
        """
        if self.krn_err_y_ is None:
            return
        self.y.unmap()
        self.err_y.unmap()
        ev = self.execute_kernel(self.krn_err_y_,
                                 [self.err_y.v.size], None)
        ev.wait()

    def cpu_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.cpu_err_y_update()
        self.cpu_err_h_update()
        self.cpu_weights_update()
        self.print_times(t1)

    def ocl_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_y_update()
        self.gpu_err_h_update()
        self.gpu_weights_update()
        self.print_times(t1)


class GDSM(GD):
    """Gradient Descent for softmax.

    It is the same as GD inside.
    """
    pass


class GDTanh(GD):
    """Gradient Descent for f(x) = 1.7159 * tanh(0.6666 * s), s = (W * x + b),
       y = a * tanh(b * s).

    f'(s) = (a * tanh(b * s))' = a * tanh'(b * s) * b
          = a * (1.0 - tanh^2(b * s)) * b
          = a * b - a * b * tanh^2(b * s)
          = a * b - y * y * b / a
          = y * y * (-b / a) + (a * b)
          = y * y * (-0.388484177) + 1.14381894
    """
    def cpu_err_y_update(self):
        """Multiply err_y by activation derivative by s in terms of y.
        """
        self.y.map_read()
        self.err_y.map_write()
        y = self.y.v
        self.err_y.v *= y * y * (-0.388484177) + 1.14381894

    def initialize(self):
        self.cl_sources_["gradient_descent_tanh.cl"] = {}
        super(GDTanh, self).initialize()
        if self.device is None:
            return
        self.krn_err_y_ = self.get_kernel("err_y_update")
        self.krn_err_y_.set_arg(0, self.err_y.v_)
        self.krn_err_y_.set_arg(1, self.y.v_)


class GDRELU(GD):
    """Gradient Descent for f(x) = log(1.0 + exp(s)), s = (W * x + b),
       y = log(1.0 + exp(s)).

    f'(s) = 1.0 / (1.0 + exp(-s)) = 1.0 - exp(-y)
    """
    def cpu_err_y_update(self):
        """Multiply err_y by activation derivative by s in terms of y.
        """
        self.y.map_read()
        self.err_y.map_write()
        y = self.y.v
        self.err_y.v *= 1.0 - numpy.exp(-y)

    def initialize(self):
        self.cl_sources_["gradient_descent_relu.cl"] = {}
        super(GDRELU, self).initialize()
        if self.device is None:
            return
        self.krn_err_y_ = self.get_kernel("err_y_update")
        self.krn_err_y_.set_arg(0, self.err_y.v_)
        self.krn_err_y_.set_arg(1, self.y.v_)
