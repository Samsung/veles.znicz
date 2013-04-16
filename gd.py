"""
Created on Apr 15, 2013

Gradient Descent Filters.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import formats
import pyopencl
import numpy


class GDSM(filters.OpenCLFilter):
    """Gradient Descent for softmax.
    
    Attributes:
        weights: weights of the current layer.
        this_errs: errors of the output values at the current layer.
        prev_errs: errors of the output values at the previous layer.
    """
    def __init__(self, device = None, unpickling = 0):
        super(GDSM, self).__init__(device=device, unpickling=unpickling)
        if unpickling:
            return
        self.weights = formats.Vector(device)
        self.this_errs = formats.Batch(device)
        self.prev_errs = formats.Batch(device)

    def initialize(self):
        if self.prev_errs.batch == None or self.prev_errs.batch.size != self.this_errs.batch.size:
            self.prev_errs.batch = filters.aligned_zeros(self.this_errs.batch.shape)
            self.prev_errs.batch_ = None

        if not self.device:
            return

        mf = pyopencl.mem_flags
        if self.prev_errs.batch_ == None:
            self.prev_errs.batch_ = pyopencl.Buffer(self.device.context_, mf.READ_WRITE | mf.USE_HOST_PTR, \
                                               hostbuf=self.prev_errs.batch)

    def run(self):
        pass
