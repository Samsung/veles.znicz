"""
Created on Apr 15, 2013

Gradient Descent Filters.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import formats
import numpy
import time
import pyopencl


class GD(filters.OpenCLFilter):
    """Gradient Descent.

    Attributes:
        weights: weights of the current layer.
        bias: bias of the current layer.
        y: outputs of the current layer.
        h: outputs of the hidden layer.
        err_y: backpropagation errors for y.
        err_h: backpropagation errors for h (will compute its).
        global_alpha: gradient descent speed (positive).
        global_lambda: coefficient (positive or zero) for weights regularization term (lambda/2 * sum(weights^2)).
        prg_: OpenCL program.
        krn_err_h_: OpenCL kernel for matrix multiplication.
        krn_weights_: OpenCL kernel for weights update.
        krn_err_y_: OpenCL kernel for err_y update.
        krn_bias_: OpenCL kernel for bias update.
        cl_sources: OpenCL source files.
    """
    def __init__(self, device = None, global_alpha = 0.9, global_lambda = 0.0000001, unpickling = 0):
        super(GD, self).__init__(device=device, unpickling=unpickling)
        self.prg_ = None
        self.krn_err_h_ = None
        self.krn_weights_ = None
        self.krn_err_y_ = None
        self.krn_bias_ = None
        self.cl_sources = ["cl/gd.cl"]
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
        if self.err_h.batch == None or self.err_h.batch.size != self.h.batch.size:
            self.err_h.batch = filters.aligned_zeros(self.h.batch.shape)
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
                       "#define Y %d\n\n") % \
                       (self.device.info.BLOCK_SIZE, self.err_h.aligned_.shape[0], \
                        self.err_h.aligned_.size // self.err_h.aligned_.shape[0], \
                        self.err_y.aligned_.size // self.err_y.aligned_.shape[0])
            s = defines
            for src in self.cl_sources:
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            fout = open("cache/gd_%d_%d.cl" % (self.h.batch.size // self.h.batch.shape[0], \
                                               self.y.batch.size // self.y.batch.shape[0]), "w")
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
        weights *= 1.0 + ((-self.global_alpha) * self.global_lambda)  # regularization (will not regularize bias)
        for i in range(0, batch_size):  # loop by batch
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            h = self.h.batch[i]
            h = h.reshape(h.size)  # make it plain
            weights += numpy.outer(h, err_y) * ((-self.global_alpha) * r_batch_size)
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
        global_size = [self.h.aligned_.size // self.h.aligned_.shape[0], \
                       self.err_y.aligned_.size // self.err_y.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        ev1 = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_weights_, global_size, local_size)
        
        self.krn_bias_.set_arg(2, kr[0])
        global_size = [self.err_y.aligned_.size // self.err_y.aligned_.shape[0], self.device.info.BLOCK_SIZE]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        ev2 = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_bias_, global_size, local_size)
        
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
        err_y = self.err_y.batch.reshape([self.err_y.batch.shape[0], \
                                          self.err_y.batch.size // self.err_y.batch.shape[0]])
        err_h = self.err_h.batch.reshape([self.err_h.batch.shape[0], \
                                          self.err_h.batch.size // self.err_h.batch.shape[0]])
        numpy.dot(err_y, weights, err_h)
        self.err_h.update()

    def gpu_err_h_update(self):
        """Backpropagate error (will compute err_h).
        """
        self.err_y.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        global_size = [self.err_h.aligned_.size // self.err_h.aligned_.shape[0], self.err_h.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_err_h_, global_size, local_size)
        event.wait()
        self.err_h.update(formats.GPU)

    def print_times(self, t_start):
        if not __debug__:
            print("Backprop within %.2f sec: %d_%d" % \
                  (time.time() - t_start, self.h.batch.size // self.h.batch.shape[0], \
                   self.y.batch.size // self.y.batch.shape[0]))
            return
        self.weights.sync()
        self.bias.sync()
        weights = self.weights.v
        bias = self.bias.v
        print("Backprop within %.2f sec: (W, b) = (%.6f, %.6f), (%.6f, %.6f)" % \
              (time.time() - t_start, weights.min(), weights.max(), bias.min(), bias.max()))

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
        global_size = [self.err_y.aligned_.size // self.err_y.aligned_.shape[0], self.err_y.aligned_.shape[0]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_err_y_, global_size, None)
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

    f'(y) = (a * tanh(b * y))' = a * (1 - b^2 * y^2) * b = a * b - a * b^3 * y^2
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
        self.cl_sources.append("cl/gd_tanh.cl")
        retval = super(GDTanh, self).initialize()
        if retval or not self.device:
            return retval
        self.krn_err_y_ = pyopencl.Kernel(self.prg_, "err_y_update")
        self.krn_err_y_.set_arg(0, self.err_y.batch_)
        self.krn_err_y_.set_arg(1, self.y.batch_)
