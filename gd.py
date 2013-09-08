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
import sys


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
    """
    def __init__(self, device=None, global_alpha=0.01, global_lambda=0.00005,
                 weights_transposed=False, store_gradient=False):
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
            self.err_h.v = numpy.zeros(self.h.v.shape,
                                           dtype=config.dtypes[config.dtype])
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

        if not self.device:
            return

        if self.prg_ == None:
            defines = ("#define APPLY_GRADIENT\n"
                       "%s\n"
                       "%s\n"
                       "%s\n"
                       "#define BLOCK_SIZE %d\n"
                       "#define BATCH %d\n"
                       "#define H %d\n"
                       "#define Y %d\n\n") % (
                    "#define WEIGHTS_TRANSPOSED"
                    if self.weights_transposed else "",
                    "#define STORE_GRADIENT"
                    if self.store_gradient else "",
                    config.cl_defines[config.dtype],
                    self.device.info.BLOCK_SIZE[config.dtype],
                    self.err_h.aligned_.shape[0],
                    self.err_h.aligned_.size // self.err_h.aligned_.shape[0],
                    self.err_y.aligned_.size // self.err_y.aligned_.shape[0])
            s = defines
            for src, define in self.cl_sources_.items():
                s += "\n" + define + "\n"
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            fin = open("%s/matrix_multiplication.cl" % (config.cl_dir), "r")
            s_mx_mul = fin.read()
            fin.close()
            s = s.replace("MX_MUL", s_mx_mul)
            fout = open("%s/gd_%d_%d.cl" % (config.cache_dir,
                self.h.v.size // self.h.v.shape[0],
                self.y.v.size // self.y.v.shape[0]), "w")
            fout.write(s)
            fout.close()

            self.prg_ = pyopencl.Program(self.device.context_, s).build()

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
        self.h.sync()
        self.err_y.sync()
        self.weights.sync()
        self.bias.sync()

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

        self.weights.update()
        self.bias.update()

    def gpu_weights_update(self):
        self.h.sync(formats.GPU)
        self.err_y.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        self.bias.sync(formats.GPU)

        batch_size = self.y.v.shape[0] if self.batch_size == None \
                                           else self.batch_size[0]
        self.cl_const[0] = -self.global_alpha / batch_size
        self.cl_const[1] = -self.global_alpha * self.global_lambda
        self.krn_weights_.set_arg(4, self.cl_const[0])
        self.krn_weights_.set_arg(5, self.cl_const[1])
        if self.weights_transposed:
            global_size = [
                self.err_y.aligned_.size // self.err_y.aligned_.shape[0],
                self.h.aligned_.size // self.h.aligned_.shape[0]]
        else:
            global_size = [
                self.h.aligned_.size // self.h.aligned_.shape[0],
                self.err_y.aligned_.size // self.err_y.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE[config.dtype],
                      self.device.info.BLOCK_SIZE[config.dtype]]
        ev1 = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                    self.krn_weights_, global_size, local_size)

        self.krn_bias_.set_arg(3, self.cl_const[0])
        global_size = [self.err_y.aligned_.size //
                       self.err_y.aligned_.shape[0],
                       self.device.info.BLOCK_SIZE[config.dtype]]
        local_size = [self.device.info.BLOCK_SIZE[config.dtype],
                      self.device.info.BLOCK_SIZE[config.dtype]]
        ev2 = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                               self.krn_bias_, global_size,
                                               local_size)

        ev1.wait()
        ev2.wait()

        self.weights.update(formats.GPU)
        self.bias.update(formats.GPU)

    def cpu_err_h_update(self):
        """Backpropagate error (will compute err_h).
        """
        self.err_y.sync()
        self.weights.sync()
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
        self.err_h.update()

    def gpu_err_h_update(self):
        """Backpropagate error (will compute err_h).
        """
        self.err_y.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        global_size = [self.err_h.aligned_.size //
                       self.err_h.aligned_.shape[0],
                       self.err_h.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE[config.dtype],
                      self.device.info.BLOCK_SIZE[config.dtype]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
            self.krn_err_h_, global_size, local_size)
        event.wait()
        self.err_h.update(formats.GPU)

    def print_times(self, t_start):
        log = self.log()
        if not log.isEnabledFor(logging.DEBUG):
            return
        self.weights.sync()
        self.bias.sync()
        weights = self.weights.v
        bias = self.bias.v
        self.log().debug("BP %d_%d in %.2f sec (min, avg, max):\t"
                         "W=%.6f,%.4f,%.2f\tB=%.6f,%.4f,%.2f" %
                  (self.h.v.size // self.h.v.shape[0],
                   self.y.v.size // self.y.v.shape[0],
                   time.time() - t_start,
                   numpy.fabs(weights).min(),
                   numpy.average(numpy.fabs(weights)),
                   numpy.fabs(weights).max(),
                   numpy.fabs(bias).min(),
                   numpy.average(numpy.fabs(bias)),
                   numpy.fabs(bias).max()))

    def cpu_err_y_update(self):
        """Multiply err_y by activation derivative by y.
        """
        pass

    def gpu_err_y_update(self):
        """Multiply err_y by activation derivative by y.
        """
        if self.krn_err_y_ == None:
            return
        self.y.sync(formats.GPU)
        self.err_y.sync(formats.GPU)
        global_size = [self.err_y.aligned_.size //
                       self.err_y.aligned_.shape[0],
                       self.err_y.aligned_.shape[0]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                                 self.krn_err_y_, global_size,
                                                 None)
        event.wait()
        self.err_y.update(formats.GPU)

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
    """Gradient Descent for f(): y = 1.7159 * tanh(0.6666 * (W * x + b)).

    f'(y) = (a * tanh(b * y))' = a * (1 - b^2 * y^2) * b
          = a * b - a * b^3 * y^2
          = 1.143819 - 0.508262 * y^2
    """
    def cpu_err_y_update(self):
        """Multiply err_y by activation derivative by y.
        """
        self.y.sync()
        self.err_y.sync()
        y = self.y.v
        self.err_y.v *= y * y * (-0.508262) + 1.143819
        self.err_y.update()

    def initialize(self):
        self.cl_sources_["%s/gradient_descent_tanh.cl" % (config.cl_dir)] = ""
        retval = super(GDTanh, self).initialize()
        if retval or not self.device:
            return retval
        self.krn_err_y_ = pyopencl.Kernel(self.prg_, "err_y_update")
        self.krn_err_y_.set_arg(0, self.err_y.v_)
        self.krn_err_y_.set_arg(1, self.y.v_)
