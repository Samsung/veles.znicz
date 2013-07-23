"""
Created on Jul 22, 2013

RBM unit.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import formats
import numpy
import pyopencl
import time
import rnd
import config
import logging
import gd


class RBM(units.OpenCLUnit):
    """RBM with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).

    Should be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        weights
        bias

    Attributes:
        input: input as Batch.
        output: output as Batch.
        weights: weights as Vector.
        bias: bias as Vector.
        output_shape: shape of the output layer.
        weights_amplitude: amplitude of the random distribution of weights.
        rand: rnd.Rand() object.
        krn_: OpenCL kernel for forward propagation without random.
        krn_apply_rand_: OpenCL kernel which applies random.
        s_activation: activation define for OpenCL source.
        output_rand: vector of random values in the shape of output.
    """
    def __init__(self, output_shape=None, device=None, weights_amplitude=0.05,
                 rand=rnd.default, unpickling=0):
        super(RBM, self).__init__(unpickling=unpickling, device=device)
        self.krn_ = None
        self.krn_apply_rand_ = None
        if unpickling:
            return
        self.cl_sources["%s/forward.cl" % (config.cl_dir,)] = ""
        self.cl_sources["%s/rbm.cl" % (config.cl_dir,)] = ""
        self.input = None  # formats.Batch(device)
        self.output = formats.Batch(device)
        self.weights = formats.Vector(device)
        self.bias = formats.Vector(device)
        self.output_shape = output_shape
        self.weights_amplitude = weights_amplitude
        self.rand = rand
        self.s_activation = "ACTIVATION_TANH"
        self.output_rand = formats.Batch(device)
        self.y_low_high = numpy.array([-1.0, 1.0],
                                      dtype=config.dtypes[config.dtype])

    def initialize(self):
        if self.weights_amplitude == None:
            self.weights_amplitude = 9.0 / (self.input.batch.size //
                                            self.input.batch.shape[0])
        n_weights = self.input.batch.size // self.input.batch.shape[0] * \
                    numpy.prod(self.output_shape)
        if self.weights.v == None or self.weights.v.size != n_weights:
            self.weights.v = numpy.zeros([n_weights],
                                         dtype=config.dtypes[config.dtype])
            self.rand.fill(self.weights.v, -self.weights_amplitude,
                           self.weights_amplitude)
            # Reshape weights as a transposed matrix:
            self.weights.v = self.weights.v.\
                reshape([numpy.prod(self.output_shape),
                         self.input.batch.size // self.input.batch.shape[0]])
            self.weights.v_ = None
        if self.bias.v == None or \
           self.bias.v.size != numpy.prod(self.output_shape):
            self.bias.v = numpy.zeros([numpy.prod(self.output_shape)],
                                      dtype=config.dtypes[config.dtype])
            self.rand.fill(self.bias.v, -self.weights_amplitude,
                           self.weights_amplitude)
            self.bias.v_ = None

        output_size = self.input.batch.shape[0] * numpy.prod(self.output_shape)
        if (self.output.batch == None or
            self.output.batch.size != output_size):
            self.output.batch = numpy.zeros([self.input.batch.shape[0],
                                        numpy.prod(self.output_shape)],
                                        dtype=config.dtypes[config.dtype])
            self.output.batch_ = None
        if (self.output_rand.batch == None or
            self.output_rand.batch.size != output_size):
            self.output_rand.batch = numpy.zeros([self.input.batch.shape[0],
                                        numpy.prod(self.output_shape)],
                                        dtype=config.dtypes[config.dtype])
            self.output_rand.batch_ = None

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)
        self.output_rand.initialize(self.device)

        if not self.device:
            return

        if self.krn_ == None:
            output_size = self.output.aligned_.size // \
                          self.output.aligned_.shape[0]
            defines = ("%s\n"
                       "#define %s\n"
                       "#define BLOCK_SIZE %d\n"
                       "#define H %d\n"
                       "#define Y %d\n"
                       "#define Y_REAL %d\n"
                       "#define BATCH %d\n\n" %
                       (config.cl_defines[config.dtype], self.s_activation,
                        self.device.info.BLOCK_SIZE[config.dtype],
                        self.weights.aligned_.size // output_size, output_size,
                        self.output.batch.size // self.output.batch.shape[0],
                        self.output.aligned_.shape[0]))
            s = defines
            for src, define in self.cl_sources.items():
                s += "\n" + define + "\n"
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            global this_dir
            fin = open("%s/matrix_multiplication.cl" % (config.cl_dir,), "r")
            s_mx_mul = fin.read()
            fin.close()
            s = s.replace("MX_MUL", s_mx_mul)
            fout = open("%s/rbm_%d_%d.cl" % (config.cache_dir,
                self.input.batch.size // self.input.batch.shape[0],
                self.output.batch.size // self.output.batch.shape[0]), "w")
            fout.write(s)
            fout.close()

            self.prg_ = pyopencl.Program(self.device.context_, s).build()

            self.krn_ = pyopencl.Kernel(self.prg_, "FEED_LAYER")
            self.krn_.set_arg(0, self.input.batch_)
            self.krn_.set_arg(1, self.weights.v_)
            self.krn_.set_arg(2, self.output.batch_)
            self.krn_.set_arg(3, self.bias.v_)

            self.krn_apply_rand_ = pyopencl.Kernel(self.prg_, "apply_rand")
            self.krn_apply_rand_.set_arg(0, self.output.batch_)
            self.krn_apply_rand_.set_arg(1, self.output_rand.batch_)

    def print_times(self, t_start):
        """Show some statistics.
        """
        log = self.log()
        if not log.isEnabledFor(logging.DEBUG):
            return
        y = self.output.batch
        self.output.sync()
        self.weights.sync()
        self.log().info("%s: %d samples with %d weights in %.2f sec "
            "(min,avg,max,sum):\ty=%.6f,%.4f,%.2f,%.2f" %
            (self.__class__.__name__, y.shape[0],
             self.weights.v.size, time.time() - t_start,
             numpy.fabs(y).min(), numpy.average(numpy.fabs(y)),
             numpy.fabs(y).max(), y.sum()))

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        self.input.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        self.bias.sync(formats.GPU)
        output_size = int(self.output.aligned_.size //
                          self.output.aligned_.shape[0])
        global_size = [output_size, self.output.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE[config.dtype],
                      self.device.info.BLOCK_SIZE[config.dtype]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_,
                                                 global_size, local_size)
        self.rand.fill(self.output_rand.batch, -1.7159, 1.7159)
        self.output_rand.update()
        self.output_rand.sync(formats.GPU)
        event.wait()
        self.krn_apply_rand_.set_arg(2, self.y_low_high[0])
        self.krn_apply_rand_.set_arg(3, self.y_low_high[1])
        global_size = [self.output.aligned_.size //
                       self.output.aligned_.shape[0],
                       self.output.aligned_.shape[0]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                    self.krn_apply_rand_, global_size, None)
        event.wait()
        self.output.update(formats.GPU)

    def cpu_run(self):
        return self.gpu_run()

    def run(self):
        t1 = time.time()
        retval = super(RBM, self).run()
        if retval:
            return retval
        self.print_times(t1)


class GDTanh(gd.GD):
    """Gradient Descent for f(): y = 1.7159 * tanh(0.6666 * (W * x + b)).

    f'(y) = (a * tanh(b * y))' = a * (1 - b^2 * y^2) * b
          = a * b - a * b^3 * y^2
          = 1.143819 - 0.508262 * y^2

    With respect to random activation.

    Attributes:
        rnd_window_size: size for applying derivative.
    """
    def __init__(self, device=None, global_alpha=0.001, global_lambda=0.00005,
                 rnd_window_size=0.1, unpickling=0):
        super(GDTanh, self).__init__(device=device, global_alpha=global_alpha,
                            global_lambda=global_lambda, unpickling=unpickling)
        if unpickling:
            return
        self.rnd_window_size = numpy.array([rnd_window_size],
                                    dtype=config.dtypes[config.dtype])
        self.y_rand = None

    def cpu_err_y_update(self):
        return self.gpu_err_y_update()

    def initialize(self):
        self.cl_sources["%s/rbm.cl" % (config.cl_dir,)] = ""
        retval = super(GDTanh, self).initialize()
        if retval or not self.device:
            return retval
        self.y_rand.initialize(self.device)
        self.krn_err_y_ = pyopencl.Kernel(self.prg_, "err_y_update")
        self.krn_err_y_.set_arg(0, self.err_y.batch_)
        self.krn_err_y_.set_arg(1, self.y.batch_)
        self.krn_err_y_.set_arg(2, self.y_rand.batch_)

    def gpu_err_y_update(self):
        self.krn_err_y_.set_arg(3, self.rnd_window_size[0])
        return super(GDTanh, self).gpu_err_y_update()
