"""
Created on Apr 15, 2013

Gradient Descent Units.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import formats
import numpy
import time
import pyopencl
import config
import logging


class GD(units.OpenCLUnit):
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
        y: outputs of the current layer.
        h: outputs of the hidden layer.
        err_y: backpropagation errors for y.
        err_h: backpropagation errors for h (will compute its).
        global_alpha: gradient descent speed (positive).
        global_lambda: coefficient (positive or zero) for weights
                       regularization term (lambda/2 * sum(weights^2)).
        krn_err_h_: OpenCL kernel for matrix multiplication.
        krn_weights_: OpenCL kernel for weights update.
        krn_err_y_: OpenCL kernel for err_y update.
        krn_bias_: OpenCL kernel for bias update.
        batch_size: effective batch size (if None, get it from y).
        weights_transposed: assume weights matrix as a transposed one.
        store_gradient: will save gradient as separate Vector().
        apply_gradient: will apply gradient.
    """
    def __init__(self, device=None, global_alpha=0.01, global_lambda=0.00005,
                 weights_transposed=False, store_gradient=False,
                 apply_gradient=True):
        super(GD, self).__init__(device=device)
        self.weights_transposed = weights_transposed
        self.weights = None  # formats.Vector(device)
        self.bias = None  # formats.Vector(device)
        self.y = None  # formats.Vector(device)
        self.h = None  # formats.Vector(device)
        self.err_y = None  # formats.Vector(device)
        self.err_h = formats.Vector(device)
        self.global_alpha = global_alpha
        self.global_lambda = global_lambda
        self.batch_size = None  # [0]
        self.gradient_weights = formats.Vector()
        self.gradient_bias = formats.Vector()
        self.store_gradient = store_gradient
        self.apply_gradient = apply_gradient
        self.cl_const = numpy.zeros(2, dtype=config.dtypes[config.dtype])

    def init_unpickled(self):
        super(GD, self).init_unpickled()
        self.cl_sources_["%s/gradient_descent.cl" % (config.cl_dir)] = ""
        self.krn_err_h_ = None
        self.krn_weights_ = None
        self.krn_err_y_ = None
        self.krn_bias_ = None

    def initialize(self):
        if (self.err_h.v == None or
            self.err_h.v.size != self.h.v.size):
            self.err_h.v = numpy.zeros(self.h.v.shape, dtype=self.h.v.dtype)
            self.err_h.v_ = None

        if (self.store_gradient and
            (self.gradient_weights.v == None or
             self.gradient_weights.v.size != self.weights.v.size)):
            self.gradient_weights.v = numpy.zeros_like(self.weights.v)
            self.gradient_weights.v_ = None

        if (self.store_gradient and
            (self.gradient_bias.v == None or
             self.gradient_bias.v.size != self.bias.v.size)):
            self.gradient_bias.v = numpy.zeros_like(self.bias.v)
            self.gradient_bias.v_ = None

        self.weights.initialize(self.device)
        self.bias.initialize(self.device)
        self.y.initialize(self.device)
        self.h.initialize(self.device)
        self.err_y.initialize(self.device)
        self.err_h.initialize(self.device)
        if self.store_gradient:
            self.gradient_weights.initialize(self.device)
            self.gradient_bias.initialize(self.device)

        if self.device == None:
            return

        if self.prg_ == None:
            defines = ("%s\n"
                       "%s\n"
                       "%s\n"
                       "%s\n"
                       "#define BLOCK_SIZE %d\n"
                       "#define BATCH %d\n"
                       "#define H %d\n"
                       "#define Y %d\n\n") % (
                    "#define APPLY_GRADIENT"
                    if self.apply_gradient else "",
                    "#define WEIGHTS_TRANSPOSED"
                    if self.weights_transposed else "",
                    "#define STORE_GRADIENT"
                    if self.store_gradient else "",
                    config.cl_defines[config.c_dtype],
                    self.device.info.BLOCK_SIZE[config.c_dtype],
                    self.err_h.v.shape[0],
                    self.err_h.v.size // self.err_h.v.shape[0],
                    self.err_y.v.size // self.err_y.v.shape[0])
            self.build_program(defines, "%s/gd_%d_%d.cl" % (config.cache_dir,
                self.h.v.size // self.h.v.shape[0],
                self.y.v.size // self.y.v.shape[0]))

            self.krn_err_h_ = pyopencl.Kernel(self.prg_, "err_h_update")
            self.krn_err_h_.set_arg(0, self.err_y.v_)
            self.krn_err_h_.set_arg(1, self.weights.v_)
            self.krn_err_h_.set_arg(2, self.err_h.v_)

            self.krn_weights_ = pyopencl.Kernel(self.prg_, "weights_update")
            self.krn_weights_.set_arg(0, self.err_y.v_)
            self.krn_weights_.set_arg(1, self.h.v_)
            self.krn_weights_.set_arg(2, self.weights.v_)
            self.krn_weights_.set_arg(3, self.gradient_weights.v_)

            self.krn_bias_ = pyopencl.Kernel(self.prg_, "bias_update")
            self.krn_bias_.set_arg(0, self.err_y.v_)
            self.krn_bias_.set_arg(1, self.bias.v_)
            self.krn_bias_.set_arg(2, self.gradient_bias.v_)

    def cpu_weights_update(self):
        self.h.map_read()
        self.err_y.map_read()
        self.weights.map_write()
        self.bias.map_write()

        batch_size = (self.y.v.shape[0] if self.batch_size == None
                                        else self.batch_size[0])

        alpha_batch = -self.global_alpha / batch_size
        alpha_lambda = -self.global_alpha * self.global_lambda

        err_y = formats.reshape(self.err_y.v,
            [self.err_y.v.shape[0],
             self.err_y.v.size // self.err_y.v.shape[0]])
        h = formats.reshape(self.h.v,
            [self.h.v.shape[0],
             self.h.v.size // self.h.v.shape[0]])
        gradient = numpy.dot(err_y.transpose(), h)
        gradient *= alpha_batch
        gradient += self.weights.v * alpha_lambda
        if self.store_gradient:
            self.gradient_weights.v[:] = gradient[:]
        if self.weights_transposed:
            self.weights.v += gradient.transpose()
        else:
            self.weights.v += gradient

        gradient = err_y.sum(axis=0) * alpha_batch
        if self.store_gradient:
            self.gradient_bias.v[:] = gradient[:]
        self.bias.v += gradient

    def gpu_weights_update(self):
        self.h.unmap()
        self.err_y.unmap()
        self.weights.unmap()
        self.bias.unmap()

        batch_size = self.y.v.shape[0] if self.batch_size == None \
                                           else self.batch_size[0]
        self.cl_const[0] = -self.global_alpha / batch_size
        self.cl_const[1] = -self.global_alpha * self.global_lambda
        self.krn_weights_.set_arg(4, self.cl_const[0])
        self.krn_weights_.set_arg(5, self.cl_const[1])
        block_size = self.device.info.BLOCK_SIZE[config.c_dtype]
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
        ev1 = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
            self.krn_weights_, global_size, local_size)

        self.krn_bias_.set_arg(3, self.cl_const[0])
        global_size = [(self.err_y.v.size // self.err_y.v.shape[0]) *
                       block_size]
        local_size = [block_size]
        ev2 = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                               self.krn_bias_, global_size,
                                               local_size)

        ev1.wait()
        ev2.wait()

    def cpu_err_h_update(self):
        """Backpropagate error (will compute err_h).
        """
        self.err_h.map_invalidate()
        self.err_y.map_read()
        self.weights.map_read()
        err_y = formats.reshape(self.err_y.v,
            [self.err_y.v.shape[0],
             self.err_y.v.size // self.err_y.v.shape[0]])
        err_h = formats.reshape(self.err_h.v,
            [self.err_h.v.shape[0],
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
        block_size = self.device.info.BLOCK_SIZE[config.c_dtype]
        global_size = [formats.roundup(self.err_h.v.size //
                       self.err_h.v.shape[0], block_size),
                       formats.roundup(self.err_h.v.shape[0],
                                       block_size)]
        local_size = [block_size, block_size]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
            self.krn_err_h_, global_size, local_size)
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
            self.log().debug("BP %d_%d in %.2f sec: min avg max: "
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
            self.log().debug("BP %d_%d in %.2f sec: min avg max: "
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
        if self.krn_err_y_ == None:
            return
        self.y.unmap()
        self.err_y.unmap()
        global_size = [self.err_y.v.size // self.err_y.v.shape[0],
                       self.err_y.v.shape[0]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                                 self.krn_err_y_, global_size,
                                                 None)
        event.wait()

    def cpu_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.cpu_err_y_update()
        self.cpu_err_h_update()
        self.cpu_weights_update()
        self.print_times(t1)

    def gpu_run(self):
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
        """Multiply err_y by activation derivative by y.
        """
        self.y.map_read()
        self.err_y.map_write()
        y = self.y.v
        self.err_y.v *= y * y * (-0.388484177) + 1.14381894

    def initialize(self):
        self.cl_sources_["%s/gradient_descent_tanh.cl" % (config.cl_dir)] = ""
        super(GDTanh, self).initialize()
        if self.device == None:
            return
        self.krn_err_y_ = pyopencl.Kernel(self.prg_, "err_y_update")
        self.krn_err_y_.set_arg(0, self.err_y.v_)
        self.krn_err_y_.set_arg(1, self.y.v_)
