"""
Created on Mar 20, 2013

All2All filters.

TODO(a.kazantsev): implement analigned matrix sizes in filters by expanding them.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import numpy
import pyopencl as cl
import opencl


class All2All(filters.GeneralFilter):
    """All2All layer to layer.

    State:
        input:
            batch: numpy array with first axis as a batch.
        output:
            batch: numpy array with first axis as a batch.
        weights:
            v: numpy array with weights.
        bias:
            v: numpy array with bias.

    Attributes:
        output_shape: shape of the output layer.
        weights_amplitude: amplitude of the default random distribution of weights.
        rand: numpy-style random generator function.
        BLOCK_SIZE: block size for matrix multiplication.
        weights_: opencl buffer for weights.
        bias_: opencl buffer for bias.
        krn_: opencl kernel handle.
    """
    def __init__(self, unpickling = 0, output_shape = [], weights_amplitude = 0.05, rand = numpy.random.rand):
        super(All2All, self).__init__(unpickling)
        self.weights_ = None
        self.bias_ = None
        self.krn_ = None
        if unpickling:
            return
        self.input = filters.State()
        self.output = filters.State()
        self.weights = filters.State()
        self.bias = filters.State()
        self.output_shape = output_shape
        self.weights_amplitude = weights_amplitude
        self.rand = rand

    def feed_from_batch(self, src):
        """Forward propagation from batch.
        """
        if not self.output.device:
            if src.output.device:
                self.output.device = src.output.device
            else:
                self.output.device = self.parent.cl.get_free_device()
        dev = self.output.device
        
        n_weights = src.output.data.size // src.output.data.shape[0] * self.output_layer_size
        if not self.weights or self.weights.size != n_weights:
            self.weights = opencl.aligned_zeros([n_weights])
            self.weights[:] = self.rand(self.weights.size)
            self.weights *= 2.0 * self.weights_amplitude
            self.weights -= self.weights_amplitude
            self.weights_ = None
        if not self.bias or self.bias.size != self.output_layer_size:
            self.bias = opencl.aligned_zeros([self.output_layer_size])
            self.bias[:] = self.rand(self.bias.size)
            self.bias *= 2.0 * self.weights_amplitude
            self.bias -= self.weights_amplitude
            self.bias_ = None

        self.output.labels = src.output.labels

        output_size = src.output.data.shape[0] * self.output_layer_size
        if not self.output.data or self.output.data.size != output_size:
            self.output.data = opencl.aligned_zeros([src.output.data.shape[0], self.output_layer_size])

        mf = cl.mem_flags
        if not src.output.data_:
            src.output.data_ = cl.Buffer(dev.context_, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=src.output.data)
        if not self.output.data_:
            self.output.data_ = cl.Buffer(dev.context_, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.output.data)
        if not self.weights_:
            self.weights_ = cl.Buffer(dev.context_, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.weights)
        if not self.bias_:
            self.bias_ = cl.Buffer(dev.context_, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.bias)

    def input_changed(self, src):
        """GeneralFilter method.
        """
        if self.mtime >= src.output.mtime:
            return
        self.mtime = src.output.mtime
        if src.output.__class__.__name__ == "DataBatch":
            return self.feed_from_batch(src)

    def feed_from_batch_ready(self):
        """When OpenCL event ready.
        """
        self.output.update_mtime()
        print("Processed %d samples with %d weights within %.2f seconds: %s" % \
              (self.output.data.shape[0], self.weights.size, self.output.mtime - self.mtime, self.__class__.__name__))
        if self.parent:
            self.parent.child_changed(self)


class All2AllTanh(All2All):
    """All2All layer to layer with scaled tanh() activation.
    """
    def feed_from_batch(self, src):
        """Forward propagation from batch. 
        """
        super(All2AllTanh, self).feed_from_batch(src)

        dev = self.output.device
        if not self.krn_:
            defines = ("#define BLOCK_SIZE %d\n"
                       "#define AB_WIDTH %d\n"
                       "#define B_HEIGHT %d\n\n") % \
                       (dev.BLOCK_SIZE, self.weights.size // self.output_layer_size, self.output_layer_size)
            fin = open("cl/feed_tanh.cl", "r")
            s = defines + fin.read()
            fin.close()
            fout = open("cache/test.cl", "w")
            fout.write(s)
            fout.close()

            prg = cl.Program(dev.context_, s).build()

            self.krn_ = cl.Kernel(prg, "FEED_LAYER")
            self.krn_.set_arg(0, src.output.data_)
            self.krn_.set_arg(1, self.weights_)
            self.krn_.set_arg(2, self.output.data_)
            self.krn_.set_arg(3, self.bias_)

        if not dev.queue_:
            dev.queue_ = cl.CommandQueue(dev.context_)

        global_size = [self.output_layer_size, self.output.data.shape[0]]
        local_size = [dev.BLOCK_SIZE, dev.BLOCK_SIZE]
        event = cl.enqueue_nd_range_kernel(dev.queue_, self.krn_, global_size, local_size)
        event.callback = self.feed_from_batch_ready
        event.callback_args = ()

        self.parent.cl.add_event(event)


class All2AllSoftmax(All2All):
    """All2All layer to layer with softmax activation.
    
    Currently, we will calculate softmax partially on cpu.
    """
    def feed_from_batch_ready(self, arr, queue_):
        arr.base.release(queue=queue_, wait_for=None)
        batch = self.output.data
        for sample in batch:
            rsum = 1.0 / sample.sum()
            sample *= rsum
        super(All2AllSoftmax, self).feed_from_batch_ready()

    def feed_from_batch(self, src):
        """Forward propagation from batch.
        """
        super(All2AllSoftmax, self).feed_from_batch(src)

        dev = self.output.device
        if not self.krn_:
            defines = ("#define BLOCK_SIZE %d\n"
                       "#define AB_WIDTH %d\n"
                       "#define B_HEIGHT %d\n\n") % \
                       (dev.BLOCK_SIZE, self.weights.size // self.output_layer_size, self.output_layer_size)
            fin = open("cl/feed_exp.cl", "r")
            s = defines + fin.read()
            fin.close()
            fout = open("cache/test.cl", "w")
            fout.write(s)
            fout.close()

            prg = cl.Program(dev.context_, s).build()

            self.krn_ = cl.Kernel(prg, "FEED_LAYER")
            self.krn_.set_arg(0, src.output.data_)
            self.krn_.set_arg(1, self.weights_)
            self.krn_.set_arg(2, self.output.data_)
            self.krn_.set_arg(3, self.bias_)

        if not dev.queue_:
            dev.queue_ = cl.CommandQueue(dev.context_)

        global_size = [self.output_layer_size, self.output.data.shape[0]]
        local_size = [dev.BLOCK_SIZE, dev.BLOCK_SIZE]
        cl.enqueue_nd_range_kernel(dev.queue_, self.krn_, global_size, local_size)

        arr, event = cl.enqueue_map_buffer(queue=dev.queue_, buf=self.output.data_, \
                flags=opencl.CL_MAP_READ, offset=0, shape=self.output.data.shape, \
                dtype=self.output.data.dtype, order="C", wait_for=None, is_blocking=False)

        event.callback = self.feed_from_batch_ready
        event.callback_args = (arr, dev.queue_)

        self.parent.cl.add_event(event)
