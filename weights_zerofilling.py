

import numpy
from zope.interface import implementer

from veles.accelerated_units import IOpenCLUnit
import veles.memory as formats
from veles.znicz.nn_units import ForwardBase


@implementer(IOpenCLUnit)
class ZeroFiller(ForwardBase):
    """Fills weights of given unit with zero on every step"""

    MAPPING = {"zero_filter"}

    def __init__(self, workflow, **kwargs):
        super(ZeroFiller, self).__init__(workflow, **kwargs)

        self.mask = formats.Vector()
        self.grouping = kwargs.get("grouping", 1)
        self.demand("weights")

    def init_unpickled(self):
        super(ZeroFiller, self).init_unpickled()
        self.sources_["weights_zerofilling"] = {}

    def initialize(self, device=None, **kwargs):
        super(ZeroFiller, self).initialize(device, **kwargs)
        if not self.weights:
            return True

        if not self.mask:
            assert len(self.weights.shape) == 2
            if (self.weights.shape[1] % self.grouping != 0 or
                    self.weights.shape[1] % self.grouping != 0):
                raise ValueError(
                    "Non-multiple of grouping weights shape detected: "
                    "%s, grouping=%d" %
                    (str(self.weights.shape), self.grouping))
            self.mask.reset(numpy.zeros_like(self.weights.mem))
            self.mask.map_invalidate()
            # TODO(a.kazantsev): add check for transposed weights.
            for kernel in range(self.weights.shape[0]):
                for chan in range(self.weights.shape[1]):
                    if kernel % self.grouping == chan % self.grouping:
                        self.mask.mem[kernel, chan] = 0
                    else:
                        self.mask.mem[kernel, chan] = 1
        else:
            assert self.mask.shape == self.weights.shape

        self.mask.initialize(device)
        self.weights.initialize(device)

    def _gpu_init(self):
        self.build_program(dtype=self.weights.dtype)

        self.assign_kernel("multiply_by_mask")
        self.set_args(self.mask, self.weights)

    def ocl_init(self):
        self._gpu_init()
        self._global_size = [self.weights.size]
        self._local_size = None

    def cuda_init(self):
        self._gpu_init()
        self._global_size = (self.weights.size, 1, 1)
        self._local_size = (1, 1, 1)

    def cpu_run(self):
        self.mask.map_read()
        self.weights.map_write()

        self.weights.mem *= self.mask.mem

    def _gpu_run(self):
        self.weights.unmap()
        self.mask.unmap()
        self.execute_kernel(self._global_size, self._local_size)

    def ocl_run(self):
        self._gpu_run()

    def cuda_run(self):
        self._gpu_run()
