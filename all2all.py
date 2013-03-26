"""
Created on Mar 20, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import numpy
import pyopencl as cl
import data_batch


class All2All(filters.GeneralFilter):
    """All2All layer to layer
    
    Attributes:
        output_layer_size: size of the output layer
        weights_amplitude: amplitude of the default random distribution of weights
        rand: numpy-style random generator function
        weights: weights
        bias: bias weights
        mtime: modification time of an input
        BLOCK_SIZE: block size for matrix multiplication
        weights_: opencl buffer for weights
        bias_: opencl buffer for bias
    """
    def __init__(self, unpickling = 0, parent = None, \
                 output_layer_size = 0, weights_amplitude = 0.05, rand = numpy.random.rand):
        super(All2All, self).__init__(unpickling, parent)
        self.weights_ = None
        self.bias_ = None
        if unpickling:
            return
        self.output = data_batch.DataBatch()
        self.output_layer_size = output_layer_size
        self.weights_amplitude = weights_amplitude
        self.rand = rand
        self.weights = None
        self.bias = None
        self.mtime = 0.0
        self.BLOCK_SIZE = 16

    def feed_from_batch(self, src):
        """Forward propagation from batch. 
        """
        n_weights = src.output.data.size // src.output.data.shape[0] * self.output_layer_size
        if not self.weights or self.weights.size != n_weights:
            self.weights = self.rand(n_weights).astype(numpy.float32)
            self.weights *= 2.0 * self.weights_amplitude
            self.weights -= self.weights_amplitude
            self.weights_ = None
        if not self.bias or self.bias.size != self.output_layer_size:
            self.bias = self.rand(self.output_layer_size).astype(numpy.float32)
            self.bias *= 2.0 * self.weights_amplitude
            self.bias -= self.weights_amplitude
            self.bias_ = None

        if not self.output.device:
            if src.output.device:
                self.output.device = src.output.device
            else:
                self.output.device = self.parent.cl.get_free_device()

        self.output.labels = src.output.labels

        output_size = src.output.data.shape[0] * self.output_layer_size
        if not self.output.data or self.output.data.size != output_size:
            self.output.data = numpy.zeros([src.output.data.shape[0], self.output_layer_size], dtype=numpy.float32)

        dev = self.output.device

        mf = cl.mem_flags
        if not src.output.data_:
            src.output.data_ = cl.Buffer(dev.context_, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src.output.data)
        if not self.output.data_:
            self.output.data_ = cl.Buffer(dev.context_, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.output.data)
        if not self.weights_:
            self.weights_ = cl.Buffer(dev.context_, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.weights)
        if not self.bias_:
            self.bias_ = cl.Buffer(dev.context_, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.bias)

    def input_changed(self, src):
        """GeneralFilter method
        """
        if self.mtime >= src.output.mtime:
            return
        self.mtime = src.output.mtime
        if src.output.__class__.__name__ == "DataBatch":
            return self.feed_from_batch(src)
        return 

    def feed_from_batch_ready(self):
        """When OpenCL event ready.
        """
        self.output.update_mtime()
        print("Processed %d samples with %d weights within %.2f seconds: %s" % \
              (self.output.data.shape[0], self.weights.size, self.output.mtime - self.mtime, self.__class__.__name__))
        if self.parent:
            self.parent.output_changed(self)


class All2AllTanh(All2All):
    """All2All layer to layer with scaled tanh() activation
    """
    def __init__(self, unpickling = 0, parent = None, \
                 output_layer_size = 0, weights_amplitude = 0.05, rand = numpy.random.rand):
        super(All2AllTanh, self).__init__(unpickling, parent, output_layer_size, weights_amplitude, rand)
        self.krn_ = None
        if unpickling:
            return

    def feed_from_batch(self, src):
        """Forward propagation from batch. 
        """
        super(All2AllTanh, self).feed_from_batch(src)

        dev = self.output.device
        if not self.krn_:
            defines = ("#define BLOCK_SIZE %d\n"
                       "#define AB_WIDTH %d\n"
                       "#define B_HEIGHT %d\n\n") % \
                       (self.BLOCK_SIZE, self.weights.size // self.output_layer_size, self.output_layer_size)
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
        local_size = [self.BLOCK_SIZE, self.BLOCK_SIZE]
        event = cl.enqueue_nd_range_kernel(dev.queue_, self.krn_, global_size, local_size)
        event.callback = self.feed_from_batch_ready
        event.callback_args = ()

        #print("enqueue_copy()...")
        #event = cl.enqueue_copy(queue, self.output.data, self.output.data_, is_blocking=False)
        #print("Done")
        
        self.parent.cl.add_event(event)
