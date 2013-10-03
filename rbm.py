"""
Created on Jul 22, 2013

RBM unit.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import all2all
import formats
import numpy
import pyopencl
import rnd
import config


class RBMTanh(all2all.All2AllTanh):
    """RBM with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).

    Attributes:
        output_rand: vector of random values in the shape of output.
        krn_apply_rand_: OpenCL kernel which applies random.
    """
    def __init__(self, output_shape=None, device=None, weights_amplitude=None,
                 rand=rnd.default, weights_transposed=False):
        super(RBMTanh, self).__init__(output_shape=output_shape, device=device,
            weights_amplitude=weights_amplitude, rand=rand,
            weights_transposed=weights_transposed)
        self.output_rand = formats.Vector(device)
        self.y_low_high = numpy.array([-1.0, 1.0],
                                      dtype=config.dtypes[config.dtype])

    def init_unpickled(self):
        super(RBMTanh, self).init_unpickled()
        self.krn_apply_rand_ = None
        self.cl_sources_["%s/rbm.cl" % (config.cl_dir)] = ""

    def initialize(self):
        retval = super(RBMTanh, self).initialize()
        if retval:
            return retval
        if (self.output_rand.v == None or
            self.output_rand.v.size != self.output.v.size):
            self.output_rand.v = numpy.zeros(self.output.v.shape,
                                             dtype=config.dtypes[config.dtype])
            self.output_rand.v_ = None
        self.output_rand.initialize(self.device)
        if not self.device:
            return
        self.krn_apply_rand_ = pyopencl.Kernel(self.prg_, "apply_rand")
        self.krn_apply_rand_.set_arg(0, self.output.v_)
        self.krn_apply_rand_.set_arg(1, self.output_rand.v_)

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        self.input.sync(formats.GPU)
        self.weights.sync(formats.GPU)
        self.bias.sync(formats.GPU)
        output_size = int(self.output.aligned_.size //
                          self.output.aligned_.shape[0])
        global_size = [output_size, self.output.aligned_.shape[0]]
        local_size = [self.device.info.BLOCK_SIZE[config.c_dtype],
                      self.device.info.BLOCK_SIZE[config.c_dtype]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_, self.krn_,
                                                 global_size, local_size)
        self.rand.fill(self.output_rand.v, -1.7159, 1.7159)
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
