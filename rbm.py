"""
Created on Jul 22, 2013

RBM unit.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy

import veles.config as config
import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.all2all as all2all


class RBMTanh(all2all.All2AllTanh):
    """RBM with scaled tanh() activation f(x) = 1.7159 * tanh(0.6666 * x).

    Attributes:
        output_rand: vector of random values in the shape of output.
        krn_apply_rand_: OpenCL kernel which applies random.
    """
    def __init__(self, workflow, **kwargs):
        super(RBMTanh, self).__init__(workflow, **kwargs)
        self.output_rand = formats.Vector()
        self.y_low_high = numpy.array([-1.0, 1.0],
                                      dtype=opencl_types.dtypes[config.dtype])

    def init_unpickled(self):
        super(RBMTanh, self).init_unpickled()
        self.krn_apply_rand_ = None
        self.cl_sources_["rbm.cl"] = ""

    def initialize(self):
        super(RBMTanh, self).initialize()
        if (self.output_rand.v is None or
            self.output_rand.v.size != self.output.v.size):
            self.output_rand.v = numpy.zeros(self.output.v.shape,
                dtype=opencl_types.dtypes[config.dtype])
            self.output_rand.v_ = None
        self.output_rand.initialize(self.device)
        if not self.device:
            return
        self.krn_apply_rand_ = self.get_kernel("apply_rand")
        self.krn_apply_rand_.set_arg(0, self.output.v_)
        self.krn_apply_rand_.set_arg(1, self.output_rand.v_)

    def gpu_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()
        self.input.unmap()
        self.weights.unmap()
        self.bias.unmap()
        output_size = int(self.output.v.size //
                          self.output.v.shape[0])
        block_size = self.device.device_info.BLOCK_SIZE[config.c_dtype]
        global_size = [formats.roundup(output_size, block_size),
                       formats.roundup(self.output.v.shape[0], block_size)]
        local_size = [block_size, block_size]
        event = self.execute_kernel(self.krn_,
                                             global_size, local_size)
        self.output_rand.map_invalidate()
        self.rand.fill_normal(self.output_rand.v, -1.7159, 1.7159)
        self.output_rand.unmap()
        event.wait()
        self.krn_apply_rand_.set_arg(2, self.y_low_high[0:1])
        self.krn_apply_rand_.set_arg(3, self.y_low_high[1:2])
        global_size = [self.output.v.size // self.output.v.shape[0],
                       self.output.v.shape[0]]
        event = self.execute_kernel(self.krn_apply_rand_, global_size, None)
        event.wait()

    def cpu_run(self):
        raise error.ErrNotImplemented()
