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


class GD(units.OpenCLUnit):
    """Gradient Descent.

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
    """
    def __init__(self, device=None, global_alpha=0.9, global_lambda=0.0,
                 unpickling=0):
        super(GD, self).__init__(device=device, unpickling=unpickling)
        self.cl_sources["cl/gd.cl"] = 1
        self.krn_err_h_ = None
        self.krn_weights_ = None
        self.krn_err_y_ = None
        self.krn_bias_ = None
        if unpickling:
            return
        self.weights = None  # formats.Vector(device)
        self.bias = None  # formats.Vector(device)
        self.y = None  # formats.Batch(device)
        self.h = None  # formats.Batch(device)
        self.err_y = None  # formats.Batch(device)
        self.err_h = formats.Batch(device)
        self.global_alpha = global_alpha
        self.global_lambda = global_lambda

    def initialize(self):
        if self.err_h.batch == None or \
           self.err_h.batch.size != self.h.batch.size:
            self.err_h.batch = numpy.zeros(self.h.batch.shape,
                                           dtype=numpy.float32)
            self.err_h.batch_ = None

        self.weights.initialize(self.device)
        self.bias.initialize(self.device)
        self.y.initialize(self.device)
        self.h.initialize(self.device)
        self.err_y.initialize(self.device)
        self.err_h.initialize(self.device)

        if not self.device:
            return

        if self.prg_ == None:
            defines = ("#define BLOCK_SIZE %d\n"
                       "#define BATCH %d\n"
                       "#define H %d\n"
                       "#define Y %d\n\n") % (self.device.info.BLOCK_SIZE,
                    self.err_h.aligned_.shape[0],
                    self.err_h.aligned_.size // self.err_h.aligned_.shape[0],
                    self.err_y.aligned_.size // self.err_y.aligned_.shape[0])
            s = defines
            for src in self.cl_sources.keys():
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            fin = open("cl/mx.cl", "r")
            s_mx_mul = fin.read()
            fin.close()
            s = s.replace("MX_MUL", s_mx_mul)
            fout = open("cache/gd_%d_%d.cl" % (self.h.batch.size //
                                               self.h.batch.shape[0],
                                               self.y.batch.size //
                                               self.y.batch.shape[0]), "w")
            fout.write(s)
            fout.close()

            self.prg_ = pyopencl.Program(self.device.context_, s).build()

            self.krn_err_h_ = pyopencl.Kernel(self.prg_, "err_h_update")
            self.krn_err_h_.set_arg(0, self.err_y.batch_)
            self.krn_err_h_.set_arg(1, self.weights.v_)
            self.krn_err_h_.set_arg(2, self.err_h.batch_)

            self.krn_weights_ = pyopencl.Kernel(self.prg_, "weights_update")
            self.krn_weights_.set_arg(0, self.err_y.batch_)
            self.krn_weights_.set_arg(1, self.h.batch_)
            self.krn_weights_.set_arg(2, self.weights.v_)

            self.krn_bias_ = pyopencl.Kernel(self.prg_, "bias_update")
            self.krn_bias_.set_arg(0, self.bias.v_)
            self.krn_bias_.set_arg(1, self.err_y.batch_)

    def cpu_weights_update(self):
        self.h.sync()
        self.err_y.sync()
        self.weights.sync()
        self.bias.sync()
        bias = self.bias.v
        bias = bias.reshape(bias.size)  # make it plain
        batch_size = self.y.batch.shape[0]
        r_batch_size = 1.0 / batch_size
        weights = self.weights.v.transpose()

        # regularization (will not regularize bias)
        weights *= 1.0 + ((-self.global_alpha) * self.global_lambda)

        for i in range(0, batch_size):  # loop by batch
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            h = self.h.batch[i]
            h = h.reshape(h.size)  # make it plain
            weights += numpy.outer(h, err_y) * ((-self.global_alpha) *
                                                r_batch_size)
            bias += err_y * ((-self.global_alpha) * r_batch_size)
        self.weights.update()
        self.bias.update()

    def gpu_weights_update(self):
        self.h.sync(formats.GPU)
        self.err_y.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        self.bias.sync(formats.GPU)

        batch_size = self.y.batch.shape[0]
        kr = numpy.empty([2], numpy.float32)
        kr[0] = (-self.global_alpha) / batch_size
        kr[1] = 1.0 + ((-self.global_alpha) * self.global_lambda)
        self.krn_weights_.set_arg(3, kr[0])
        self.krn_weights_.set_arg(4, kr[1])
        global_size = [self.h.aligned_.size // self.h.aligned_.shape[0],
                    self.err_y.aligned_.size // self.err_y.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        ev1 = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                    self.krn_weights_, global_size, local_size)

        self.krn_bias_.set_arg(2, kr[0])
        global_size = [self.err_y.aligned_.size //
                       self.err_y.aligned_.shape[0],
                       self.device.info.BLOCK_SIZE]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
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
        self.weights.sync()
        self.err_y.sync()
        weights = self.weights.v
        err_y = self.err_y.batch.reshape([self.err_y.batch.shape[0],
                                          self.err_y.batch.size //
                                          self.err_y.batch.shape[0]])
        err_h = self.err_h.batch.reshape([self.err_h.batch.shape[0],
                                          self.err_h.batch.size //
                                          self.err_h.batch.shape[0]])
        numpy.dot(err_y, weights, err_h)
        self.err_h.update()

    def gpu_err_h_update(self):
        """Backpropagate error (will compute err_h).
        """
        self.err_y.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        global_size = [self.err_h.aligned_.size //
                       self.err_h.aligned_.shape[0],
                       self.err_h.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                                 self.krn_err_h_, global_size,
                                                 local_size)
        event.wait()
        self.err_h.update(formats.GPU)

    def print_times(self, t_start):
        if not __debug__:
            #print("Backprop within %.2f sec: %d_%d" %
            #      (time.time() - t_start, self.h.batch.size //
            #       self.h.batch.shape[0],
            #       self.y.batch.size // self.y.batch.shape[0]))
            return
        self.weights.sync()
        self.bias.sync()
        weights = self.weights.v
        bias = self.bias.v
        if "weights_alphas" in self.__dict__:
            self.weights_alphas.sync()
            self.bias_alphas.sync()
            wa = self.weights_alphas.v
            ba = self.bias_alphas.v
            print("BP %d_%d in %.2f sec: (W; b; Wa; ba) = "
                  "(%.6f, %.3f; %.6f, %.3f; %.6f, %.3f; %.6f - %.3f)" %
                  (self.h.batch.size // self.h.batch.shape[0],
                   self.y.batch.size // self.y.batch.shape[0],
                   time.time() - t_start,
                   numpy.fabs(weights).min(), numpy.fabs(weights).max(),
                   numpy.fabs(bias).min(), numpy.fabs(bias).max(),
                   numpy.fabs(wa).min(), numpy.fabs(wa).max(),
                   numpy.fabs(ba).min(), numpy.fabs(ba).max()))
        else:
            print("BP  %d_%d in %.2f sec: (W; b) = "
                  "(%.6f, %.3f; %.6f, %.3f)" %
                  (self.h.batch.size // self.h.batch.shape[0],
                   self.y.batch.size // self.y.batch.shape[0],
                   time.time() - t_start,
                   numpy.fabs(weights).min(), numpy.fabs(weights).max(),
                   numpy.fabs(bias).min(), numpy.fabs(bias).max()))

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
        y = self.y.batch
        self.err_y.batch *= y * y * (-0.508262) + 1.143819
        self.err_y.update()

    def initialize(self):
        self.cl_sources["cl/gd_tanh.cl"] = 1
        retval = super(GDTanh, self).initialize()
        if retval or not self.device:
            return retval
        self.krn_err_y_ = pyopencl.Kernel(self.prg_, "err_y_update")
        self.krn_err_y_.set_arg(0, self.err_y.batch_)
        self.krn_err_y_.set_arg(1, self.y.batch_)


class GDA(GD):
    """GD with individual alphas for each weight.

    Attributes:
        alpha_inc: step for alpha increase.
        alpha_dec: step for alpha decrease.
        alpha_max: maximum value for alpha.
        alpha_min: minimum value for alpha.
        weights_alphas: alphas for weights.
        bias_alphas: alphas for bias.
        krn_weights_a_: kernel for weights and alphas update.
        krn_bias_a_: kernel for bias and alphas update.
    """
    def __init__(self, device=None, global_alpha=0.9, global_lambda=0.0,
                 alpha_inc=1.05, alpha_dec=0.7,
                 alpha_max=7.0, alpha_min=0.000001,
                 unpickling=0):
        super(GDA, self).__init__(device=device, global_alpha=global_alpha,
            global_lambda=global_lambda, unpickling=unpickling)
        self.krn_weights_a_ = None
        self.krn_bias_a_ = None
        if unpickling:
            return
        self.alpha_inc = alpha_inc
        self.alpha_dec = alpha_dec
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.weights_alphas = formats.Vector()
        self.bias_alphas = formats.Vector()

    def initialize(self):
        self.cl_sources["cl/gda.cl"] = 1
        retval = super(GDA, self).initialize()
        if retval:
            return retval
        if self.weights_alphas.v == None or \
           self.weights_alphas.v.size != self.weights.v.size:
            self.weights_alphas.v = numpy.zeros_like(self.weights.v)
            self.weights_alphas.v[:] = self.global_alpha
            self.weights_alphas.v_ = None
        if self.bias_alphas.v == None or \
           self.bias_alphas.v.size != self.bias.v.size:
            self.bias_alphas.v = numpy.zeros_like(self.bias.v)
            self.bias_alphas.v[:] = self.global_alpha
            self.bias_alphas.v_ = None
        self.weights_alphas.initialize(self.device)
        self.bias_alphas.initialize(self.device)
        if not self.device:
            return
        self.krn_weights_a_ = pyopencl.Kernel(self.prg_, "weights_update_a")
        self.krn_weights_a_.set_arg(0, self.err_y.batch_)
        self.krn_weights_a_.set_arg(1, self.h.batch_)
        self.krn_weights_a_.set_arg(2, self.weights.v_)
        self.krn_weights_a_.set_arg(9, self.weights_alphas.v_)

        self.krn_bias_a_ = pyopencl.Kernel(self.prg_, "bias_update_a")
        self.krn_bias_a_.set_arg(0, self.bias.v_)
        self.krn_bias_a_.set_arg(1, self.err_y.batch_)
        self.krn_bias_a_.set_arg(7, self.bias_alphas.v_)

    def gpu_weights_update(self):
        self.h.sync(formats.GPU)
        self.err_y.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        self.bias.sync(formats.GPU)
        self.weights_alphas.sync(formats.GPU)
        self.bias_alphas.sync(formats.GPU)

        batch_size = self.y.batch.shape[0]
        kr = numpy.empty([6], numpy.float32)
        kr[0] = 1.0 / batch_size
        kr[1] = self.global_lambda
        kr[2] = self.alpha_inc
        kr[3] = self.alpha_dec
        kr[4] = self.alpha_max
        kr[5] = self.alpha_min
        self.krn_weights_a_.set_arg(3, kr[0])
        self.krn_weights_a_.set_arg(4, kr[1])
        self.krn_weights_a_.set_arg(5, kr[2])
        self.krn_weights_a_.set_arg(6, kr[3])
        self.krn_weights_a_.set_arg(7, kr[4])
        self.krn_weights_a_.set_arg(8, kr[5])
        global_size = [self.h.aligned_.size // self.h.aligned_.shape[0],
                    self.err_y.aligned_.size // self.err_y.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        ev1 = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                    self.krn_weights_a_, global_size, local_size)

        self.krn_bias_a_.set_arg(2, kr[0])
        self.krn_bias_a_.set_arg(3, kr[2])
        self.krn_bias_a_.set_arg(4, kr[3])
        self.krn_bias_a_.set_arg(5, kr[4])
        self.krn_bias_a_.set_arg(6, kr[5])
        global_size = [self.err_y.aligned_.size //
                       self.err_y.aligned_.shape[0],
                       self.device.info.BLOCK_SIZE]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        ev2 = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                               self.krn_bias_a_, global_size,
                                               local_size)

        ev1.wait()
        ev2.wait()

        self.weights.update(formats.GPU)
        self.bias.update(formats.GPU)
        self.weights_alphas.update(formats.GPU)
        self.bias_alphas.update(formats.GPU)


class GDASM(GDA):
    """Gradient Descent for softmax.

    It is the same as GD inside.
    """
    pass


class GDATanh(GDA, GDTanh):
    """GDTanh with individual alphas for each weight.
    """
    pass
