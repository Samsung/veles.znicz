"""
Created on Nov 14, 2013

Gradient Descent for Convolutional Units.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import logging
import numpy
import time
import config
import error
import formats
import nn_units
import opencl_types
import znicz_config


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
        krn_err_h_clear_: OpenCL kernel for setting err_h with zeros.
        krn_err_h_: OpenCL kernel for computing err_h.
        krn_weights_: OpenCL kernel for weights update.
        krn_err_y_: OpenCL kernel for err_y update.
        krn_bias_: OpenCL kernel for bias update.
        n_kernels: number of convolutional kernels.
        kx: kernel width.
        ky: kernel height.
    """
    def __init__(self, workflow, **kwargs):
        super(GD, self).__init__(workflow, **kwargs)
        self.n_kernels = kwargs["n_kernels"]
        self.kx = kwargs["kx"]
        self.ky = kwargs["ky"]
        self.cl_const = numpy.zeros(2, dtype=opencl_types.dtypes[config.dtype])
        self.reduce_size = 64

    def init_unpickled(self):
        super(GD, self).init_unpickled()
        self.cl_sources_["gradient_descent_conv.cl"] = {}
        self.krn_err_h_clear_ = None
        self.krn_err_h_ = None
        self.krn_weights_ = None
        self.krn_err_y_ = None
        self.krn_bias_ = None

    def initialize(self):
        batch_size = self.h.v.shape[0]
        sy = self.h.v.shape[1]
        sx = self.h.v.shape[2]
        n_channels = self.h.v.size // (batch_size * sx * sy)
        n_weights = self.n_kernels * self.kx * self.ky * n_channels
        if self.weights.v.size != n_weights:
            raise error.ErrBadFormat("Expected number of weights to match "
                "input, n_kernels, kx, ky parameters")
        if self.bias.v.size != self.n_kernels:
            raise error.ErrBadFormat("Expected bias to match n_kernels")
        if self.h.v.size != batch_size * sy * sx * n_channels:
            raise error.ErrBadFormat("Expected input size to match "
                "batch_size * sy * sx * n_channels")

        if (self.err_h.v is None or
            self.err_h.v.size != self.h.v.size):
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

        if self.prg_ is None:
            block_size = self.device.device_info.BLOCK_SIZE[config.c_dtype]
            self.reduce_size = min(self.reduce_size,
                                   self.kx * self.ky * n_channels)

            defines = {
                'USE_ATOMICS': 1,
                'BLOCK_SIZE': block_size,
                'BATCH': batch_size,
                'SX': sx,
                'SY': sy,
                'N_CHANNELS': n_channels,
                'KX': self.kx,
                'KY': self.ky,
                'N_KERNELS': self.n_kernels,
                'REDUCE_SIZE': self.reduce_size
            }
            if self.apply_gradient:
                defines['APPLY_GRADIENT'] = 1
            if self.weights_transposed:
                defines['WEIGHTS_TRANSPOSED'] = 1
            if self.store_gradient:
                defines['STORE_GRADIENT'] = 1
            self.build_program(defines, "%s/gd_conv_%d_%d.cl" % (
                config.cache_dir,
                self.h.v.size // self.h.v.shape[0],
                self.y.v.size // self.y.v.shape[0]))

            self.krn_err_h_clear_ = self.get_kernel("array_clear")
            self.krn_err_h_clear_.set_arg(0, self.err_h.v_)

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

    def gpu_weights_update(self):
        self.h.unmap()
        self.err_y.unmap()
        self.weights.unmap()
        self.bias.unmap()
        self.gradient_weights.unmap()
        self.gradient_bias.unmap()

        batch_size = (self.y.v.shape[0] if self.batch_size is None
                                        else self.batch_size[0])
        sy = self.h.v.shape[1]
        sx = self.h.v.shape[2]
        n_channels = self.h.v.size // (self.h.v.shape[0] * sx * sy)
        # batch_size *= (sy - self.ky + 1) * (sx - self.kx + 1)
        self.cl_const[0] = -self.global_alpha / batch_size
        self.cl_const[1] = -self.global_alpha * self.global_lambda
        self.krn_weights_.set_arg(4, self.cl_const[0:1])
        self.krn_weights_.set_arg(5, self.cl_const[1:2])
        block_size = self.device.device_info.BLOCK_SIZE[config.c_dtype]
        if self.weights_transposed:
            global_size = [
                formats.roundup(self.n_kernels, block_size),
                formats.roundup(self.kx * self.ky * n_channels,
                                block_size)]
        else:
            global_size = [
                formats.roundup(self.kx * self.ky * n_channels,
                                block_size),
                formats.roundup(self.n_kernels, block_size)]
        local_size = [block_size, block_size]
        ev1 = self.execute_kernel(self.krn_weights_,
                                           global_size, local_size)

        self.krn_bias_.set_arg(3, self.cl_const[0:1])
        global_size = [self.n_kernels * self.reduce_size]
        local_size = [self.reduce_size]
        ev2 = self.execute_kernel(self.krn_bias_,
                                           global_size, local_size)

        ev1.wait()
        ev2.wait()

    def gpu_err_h_update(self):
        """Backpropagate error (will compute err_h).
        """
        self.err_h.unmap()
        self.err_y.unmap()
        self.weights.unmap()

        # Clear the resulting matrix
        event = self.execute_kernel(self.krn_err_h_clear_,
                                             [self.err_h.v.size], None)
        event.wait()

        batch_size = self.h.v.shape[0]
        sy = self.h.v.shape[1]
        sx = self.h.v.shape[2]
        n_channels = self.h.v.size // (batch_size * sx * sy)
        block_size = self.device.device_info.BLOCK_SIZE[config.c_dtype]
        kernel_size = self.kx * self.ky * n_channels
        global_size = [formats.roundup(kernel_size, block_size),
            formats.roundup(self.h.v.size // kernel_size, block_size)]
        local_size = [block_size, block_size]
        event = self.execute_kernel(self.krn_err_h_,
                                             global_size, local_size)
        event.wait()

    def print_times(self, t_start):
        log = self.log()
        if not log.isEnabledFor(logging.DEBUG):
            return
        self.weights.map_read()
        self.bias.map_read()
        weights = self.weights.v
        bias = self.bias.v
        if weights.dtype in (numpy.complex64, numpy.complex128):
            self.debug("BP %d_%d in %.2f sec: min avg max: "
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
            self.debug("BP %d_%d in %.2f sec: min avg max: "
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

    def gpu_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_y_update()
        self.gpu_err_h_update()
        self.gpu_weights_update()
        self.print_times(t1)

    def cpu_run(self):
        raise error.ErrNotImplemented()


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
        """Multiply err_y by activation derivative by y.
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
