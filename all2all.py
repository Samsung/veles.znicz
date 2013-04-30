"""
Created on Mar 20, 2013

All2All filters.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import formats
import numpy
import pyopencl
import time
#import matplotlib.pyplot as pp
#import matplotlib.cm as cm


class All2All(filters.OpenCLFilter):
    """All2All layer to layer.

    State:
        input: input as Batch.
        output: output as Batch.
        weights: weights as Vector.
        bias: bias as Vector.

    Attributes:
        output_shape: shape of the output layer.
        weights_amplitude: amplitude of the default random distribution of weights.
        rand: numpy-style random generator function.
        krn_: OpenCL kernel.
    """
    def __init__(self, output_shape = None, device=None, weights_amplitude = 0.05, rand = numpy.random.rand, \
                 unpickling = 0):
        super(All2All, self).__init__(unpickling=unpickling, device=device)
        self.krn_ = None
        if unpickling:
            return
        self.input = None  # formats.Batch(device)
        self.output = formats.Batch(device)
        self.weights = formats.Vector(device)
        self.bias = formats.Vector(device)
        self.output_shape = output_shape
        self.weights_amplitude = weights_amplitude
        self.rand = rand

    def show_weights(self):
        return #TODO(a.kazantsev): do properly.
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
        pp.savefig(fnme, dpi=width//8)
        print("Weights picture saved to %s" % (fnme, ))
        pp.clf()
        pp.cla()
        pp.imshow(weights, interpolation="lanczos", cmap=cm.gray)
        fnme = "cache/weights_%d_%d.png" % (numpy.prod(input_shape), output_size)
        pp.savefig(fnme, dpi=width//8)
        print("Weights picture as matrix saved to %s" % (fnme, ))
        pp.clf()
        pp.cla()

    def _initialize(self, cl_src):
        n_weights = self.input.batch.size // self.input.batch.shape[0] * numpy.prod(self.output_shape)
        if self.weights.v == None or self.weights.v.size != n_weights:
            self.weights.v = filters.aligned_zeros([n_weights])
            self.weights.v[:] = self.rand(self.weights.v.size)
            self.weights.v *= 2.0 * self.weights_amplitude
            self.weights.v -= self.weights_amplitude
            # Reshape weights as a transposed matrix:
            self.weights.v = self.weights.v.reshape([numpy.prod(self.output_shape), \
                                                     self.input.batch.size // self.input.batch.shape[0]])
            self.weights.v_ = None
        if self.bias.v == None or self.bias.v.size != numpy.prod(self.output_shape):
            self.bias.v = filters.aligned_zeros([numpy.prod(self.output_shape)])
            self.bias.v[:] = self.rand(self.bias.v.size)
            self.bias.v *= 2.0 * self.weights_amplitude
            self.bias.v -= self.weights_amplitude
            self.bias.v_ = None

        output_size = self.input.batch.shape[0] * numpy.prod(self.output_shape)
        if self.output.batch == None or self.output.batch.size != output_size:
            self.output.batch = filters.aligned_zeros([self.input.batch.shape[0], numpy.prod(self.output_shape)])
            self.output.batch_ = None

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if not self.device:
            return

        if self.krn_ == None:
            output_size = self.output.aligned_.size // self.output.aligned_.shape[0]
            defines = ("#define BLOCK_SIZE %d\n"
                       "#define AB_WIDTH %d\n"
                       "#define B_HEIGHT %d\n\n") % \
                       (self.device.info.BLOCK_SIZE, self.weights.aligned_.size // output_size, output_size)
            fin = open("cl/"+cl_src, "r")
            s = defines + fin.read()
            fin.close()
            fout = open("cache/feed_%d_%d.cl" % (self.input.batch.size // self.input.batch.shape[0], \
                                                 self.output.batch.size // self.output.batch.shape[0]), "w")
            fout.write(s)
            fout.close()

            prg = pyopencl.Program(self.device.context_, s).build()

            self.krn_ = pyopencl.Kernel(prg, "FEED_LAYER")
            self.krn_.set_arg(0, self.input.batch_)
            self.krn_.set_arg(1, self.weights.v_)
            self.krn_.set_arg(2, self.output.batch_)
            self.krn_.set_arg(3, self.bias.v_)

    def print_times(self, t_start):
        """Show some statistics.
        """
        if not __debug__:
            print("%s within %.2f sec: %d_%d" % \
                  (self.__class__.__name__, time.time() - t_start, \
                   self.input.batch.size // self.input.batch.shape[0], \
                   self.output.batch.size // self.output.batch.shape[0]))
            return
        y = self.output.batch
        self.output.sync()
        self.weights.sync()
        print("%s: %d samples with %d weights within %.2f sec: y: (min, max, sum, avg) = (%.4f, %.4f, %.4f, %.4f)" % \
              (self.__class__.__name__, y.shape[0], self.weights.v.size, time.time() - t_start, \
               y.min(), y.max(), y.sum(), numpy.average(y)))
        self.show_weights()


class All2AllTanh(All2All):
    """All2All layer to layer with scaled tanh() activation.
    """
    def initialize(self):
        self._initialize("feed_tanh.cl")

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        t1 = time.time()
        self.input.sync()
        self.weights.sync()
        self.bias.sync()
        a = self.input.batch.reshape([self.input.batch.shape[0], self.input.batch.size // self.input.batch.shape[0]])
        b = self.weights.v.transpose()
        numpy.dot(a, b, self.output.batch)
        self.output.batch[:] += self.bias.v
        self.output.batch *= 0.6666
        numpy.tanh(self.output.batch, self.output.batch)
        self.output.batch *= 1.7159
        self.output.update()
        self.print_times(t1)

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        t1 = time.time()
        self.input.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        self.bias.sync(formats.GPU)
        output_size = int(self.output.aligned_.size // self.output.aligned_.shape[0])
        global_size = [output_size, self.output.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_, global_size, local_size)
        event.wait()
        self.output.update(formats.GPU)
        self.print_times(t1)


class All2AllSoftmax(All2All):
    """All2All layer to layer with softmax activation.
    
    Currently, we will calculate softmax partially on cpu.
    """
    def initialize(self):
        self._initialize("feed_linear.cl")

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

    def cpu_run(self):
        """Forward propagation from batch on CPU only. 
        """
        t1 = time.time()
        self.input.sync()
        self.weights.sync()
        self.bias.sync()
        a = self.input.batch.reshape([self.input.batch.shape[0], self.input.batch.size // self.input.batch.shape[0]])
        b = self.weights.v.transpose()
        numpy.dot(a, b, self.output.batch)
        self.output.batch[:] += self.bias.v
        self.output.update()
        self.cpu_apply_exp()
        self.print_times(t1)

    def gpu_run(self):
        """Forward propagation from batch on GPU. 
        """
        t1 = time.time()
        self.input.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        self.bias.sync(formats.GPU)
        output_size = int(self.output.aligned_.size // self.output.aligned_.shape[0])
        global_size = [output_size, self.output.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE, self.device.info.BLOCK_SIZE]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_, global_size, local_size)
        event.wait()
        self.output.update(formats.GPU)
        self.cpu_apply_exp()
        self.print_times(t1)
