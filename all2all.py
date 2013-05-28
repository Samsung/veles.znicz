"""
Created on Mar 20, 2013

All2All units.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import formats
import numpy
import pyopencl
import time
import rnd
import config
import matplotlib.pyplot as pp
import matplotlib.cm as cm


class All2All(units.OpenCLUnit):
    """All2All with linear activation f(x) = x.

    State:
        input: input as Batch.
        output: output as Batch.
        weights: weights as Vector.
        bias: bias as Vector.

    Attributes:
        output_shape: shape of the output layer.
        weights_amplitude: amplitude of the random distribution of weights.
        rand: rnd.Rand() object.
        krn_: OpenCL kernel.
        s_activation: activation define for OpenCL source.
    """
    def __init__(self, output_shape=None, device=None, weights_amplitude=0.05,
                 rand=rnd.default, unpickling=0):
        super(All2All, self).__init__(unpickling=unpickling, device=device)
        self.krn_ = None
        if unpickling:
            return
        self.cl_sources["cl/feed.cl"] = 1
        self.input = None  # formats.Batch(device)
        self.output = formats.Batch(device)
        self.weights = formats.Vector(device)
        self.bias = formats.Vector(device)
        self.output_shape = output_shape
        self.weights_amplitude = weights_amplitude
        self.rand = rand
        self.s_activation = "ACTIVATION_LINEAR"

    def show_weights(self):
        return  # TODO(a.kazantsev): do properly.
        pp.rcParams.update({'font.size': 7})
        output_size = self.output.batch.size // self.output.batch.shape[0]
        n_cols = numpy.floor(numpy.sqrt(output_size))
        n_rows = numpy.ceil(output_size / n_cols)
        weights = self.weights.v
        input_shape = self.input.batch.shape[1:]
        print("Input shape is: %s" % (str(input_shape), ))
        for i in range(0, output_size):
            pp.subplot(n_rows, n_cols, i + 1)
            im = weights[i].reshape(input_shape)
            if len(im.shape) == 2:
                pp.imshow(im, interpolation="lanczos", cmap=cm.gray)
            else:
                im = im.reshape(im.size)
                pp.plot(im)
        width = 1024
        fnme = "cache/feed_%d_%d.png" % (numpy.prod(input_shape), output_size)
        pp.savefig(fnme, dpi=width // 8)
        print("Weights picture saved to %s" % (fnme, ))
        pp.clf()
        pp.cla()
        pp.imshow(weights, interpolation="lanczos", cmap=cm.gray)
        fnme = "cache/weights_%d_%d.png" % (numpy.prod(input_shape),
                                            output_size)
        pp.savefig(fnme, dpi=width // 8)
        print("Weights picture as matrix saved to %s" % (fnme, ))
        pp.clf()
        pp.cla()

    def initialize(self):
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
        if self.output.batch == None or self.output.batch.size != output_size:
            self.output.batch = numpy.zeros([self.input.batch.shape[0],
                                             numpy.prod(self.output_shape)],
                                            dtype=config.dtypes[config.dtype])
            self.output.batch_ = None

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if not self.device:
            return

        if self.krn_ == None:
            output_size = self.output.aligned_.size // \
                          self.output.aligned_.shape[0]
            defines = ("#define dtype %s\n"
                       "#define %s\n"
                       "#define BLOCK_SIZE %d\n"
                       "#define H %d\n"
                       "#define Y %d\n"
                       "#define Y_REAL %d\n"
                       "#define BATCH %d\n\n" %
                       (config.dtype, self.s_activation,
                        self.device.info.BLOCK_SIZE[config.dtype],
                        self.weights.aligned_.size // output_size, output_size,
                        self.output.batch.size // self.output.batch.shape[0],
                        self.output.aligned_.shape[0]))
            s = defines
            for src in self.cl_sources.keys():
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            fin = open("cl/mx.cl", "r")
            s_mx_mul = fin.read()
            fin.close()
            s = s.replace("MX_MUL", s_mx_mul)
            fout = open("cache/feed_%d_%d.cl" % (self.input.batch.size // \
                                                 self.input.batch.shape[0],
                                                 self.output.batch.size // \
                                                 self.output.batch.shape[0]),
                        "w")
            fout.write(s)
            fout.close()

            self.prg_ = pyopencl.Program(self.device.context_, s).build()

            self.krn_ = pyopencl.Kernel(self.prg_, "FEED_LAYER")
            self.krn_.set_arg(0, self.input.batch_)
            self.krn_.set_arg(1, self.weights.v_)
            self.krn_.set_arg(2, self.output.batch_)
            self.krn_.set_arg(3, self.bias.v_)

    def print_times(self, t_start):
        """Show some statistics.
        """
        if not __debug__:
            #print("%s within %.2f sec: %d_%d" % \
            #      (self.__class__.__name__, time.time() - t_start, \
            #       self.input.batch.size // self.input.batch.shape[0], \
            #       self.output.batch.size // self.output.batch.shape[0]))
            return
        y = self.output.batch
        self.output.sync()
        self.weights.sync()
        print("%s: %d samples with %d weights in %.2f sec (min,avg,max,sum):\t"
              "y=%.6f,%.4f,%.2f,%.2f" %
              (self.__class__.__name__.replace("All2All", ""), y.shape[0],
               self.weights.v.size, time.time() - t_start,
               numpy.fabs(y).min(),
               numpy.average(numpy.fabs(y)),
               numpy.fabs(y).max(),
               y.sum()))
        self.show_weights()

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        self.input.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        self.bias.sync(formats.GPU)
        output_size = int(self.output.aligned_.size // \
                          self.output.aligned_.shape[0])
        global_size = [output_size, self.output.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE[config.dtype],
                      self.device.info.BLOCK_SIZE[config.dtype]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_,
                                                 global_size, local_size)
        event.wait()
        self.output.update(formats.GPU)

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.input.sync()
        self.weights.sync()
        self.bias.sync()
        a = self.input.batch.reshape([self.input.batch.shape[0],
                                      self.input.batch.size // \
                                      self.input.batch.shape[0]])
        b = self.weights.v.transpose()
        numpy.dot(a, b, self.output.batch)
        self.output.batch[:] += self.bias.v
        self.output.update()

    def run(self):
        t1 = time.time()
        retval = super(All2All, self).run()
        if retval:
            return retval
        self.print_times(t1)


class All2AllTanh(All2All):
    """All2All with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).
    """
    def initialize(self):
        self.s_activation = "ACTIVATION_TANH"
        return super(All2AllTanh, self).initialize()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        retval = super(All2AllTanh, self).cpu_run()
        if retval:
            return retval
        self.output.sync()
        self.output.batch *= 0.6666
        numpy.tanh(self.output.batch, self.output.batch)
        self.output.batch *= 1.7159
        self.output.update()


class All2AllSoftmax(All2All):
    """All2All with linear activation and softmax normalization.

    Attributes:
        krn_sm_: kernel for softmax activation calculation.
    """
    def __init__(self, output_shape=None, device=None, weights_amplitude=0.05,
                 rand=rnd.default, unpickling=0):
        super(All2AllSoftmax, self).__init__(output_shape=output_shape,
            device=device, weights_amplitude=weights_amplitude, rand=rand,
            unpickling=unpickling)
        self.krn_sm_ = None
        if unpickling:
            return

    def initialize(self):
        self.cl_sources["cl/sm.cl"] = 1
        retval = super(All2AllSoftmax, self).initialize()
        if retval or not self.device:
            return retval
        self.krn_sm_ = pyopencl.Kernel(self.prg_, "apply_exp")
        self.krn_sm_.set_arg(0, self.output.batch_)

    def cpu_apply_exp(self):
        self.output.sync()
        if __debug__:
            s = []
            a = numpy.sort(self.output.batch.reshape(self.output.batch.size))
            for i in range(a.size - 1, a.size - 11, -1):
                s.append("%.2f" % (a[i], ))
            print("Softmax Wx+b: ", ", ".join(s), ", %.2f" % (a[0], ))
        for sample in self.output.batch:
            m = sample.max()
            sample -= m
            numpy.exp(sample, sample)
            smm = sample.sum()
            sample /= smm
        self.output.update()

    def gpu_apply_exp(self):
        self.output.sync(formats.GPU)
        global_size = [self.device.info.BLOCK_SIZE[config.dtype],
                       self.output.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE[config.dtype],
                      self.device.info.BLOCK_SIZE[config.dtype]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                                 self.krn_sm_,
                                                 global_size, local_size)
        event.wait()
        self.output.update(formats.GPU)

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        retval = super(All2AllSoftmax, self).cpu_run()
        if retval:
            return retval
        self.cpu_apply_exp()

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        retval = super(All2AllSoftmax, self).gpu_run()
        if retval:
            return retval
        self.gpu_apply_exp()
