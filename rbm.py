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
import gd


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
            self.output_rand.v = numpy.zeros_like(self.output.v)
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
        local_size = [self.device.info.BLOCK_SIZE[config.dtype],
                      self.device.info.BLOCK_SIZE[config.dtype]]
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
                 weights_transposed=False, rnd_window_size=0.1):
        super(GDTanh, self).__init__(
            device=device, global_alpha=global_alpha,
            global_lambda=global_lambda,
            weights_transposed=weights_transposed)
        self.rnd_window_size = numpy.array([rnd_window_size],
                                    dtype=config.dtypes[config.dtype])
        self.y_rand = None

    def cpu_err_y_update(self):
        return self.gpu_err_y_update()

    def initialize(self):
        self.cl_sources_["%s/rbm.cl" % (config.cl_dir)] = ""
        retval = super(GDTanh, self).initialize()
        if retval or not self.device:
            return retval
        self.y_rand.initialize(self.device)
        self.krn_err_y_ = pyopencl.Kernel(self.prg_, "err_y_update")
        self.krn_err_y_.set_arg(0, self.err_y.v_)
        self.krn_err_y_.set_arg(1, self.y.v_)
        self.krn_err_y_.set_arg(2, self.y_rand.v_)

    def gpu_err_y_update(self):
        self.krn_err_y_.set_arg(3, self.rnd_window_size[0])
        return super(GDTanh, self).gpu_err_y_update()
