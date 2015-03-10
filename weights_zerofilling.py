

import numpy
from zope.interface import implementer

from veles.accelerated_units import IOpenCLUnit
import veles.memory as formats
from veles.distributable import TriviallyDistributable
from veles.znicz.nn_units import ForwardBase


@implementer(IOpenCLUnit)
class ZeroFiller(ForwardBase, TriviallyDistributable):
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

    @property
    def effective_shape(self):
        return (self.weights.shape[0],
                self.weights.size // self.weights.shape[0])

    @property
    def grouping(self):
        return self._grouping

    @grouping.setter
    def grouping(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "grouping value must be an integer (got %s)" % type(value))
        if value < 2:
            raise ValueError("grouping value %d is invalid" % value)
        self._grouping = value

    def initialize(self, device=None, **kwargs):
        super(ZeroFiller, self).initialize(device, **kwargs)
        if not self.weights:
            return True

        if not self.mask:
            if self.effective_shape[1] % self.grouping != 0:
                raise ValueError(
                    "Non-multiple of grouping weights shape detected: "
                    "%s, grouping=%d" %
                    (self.weights.shape, self.grouping))
            self.mask.reset(numpy.zeros(self.effective_shape))
            self.mask.map_invalidate()
            # TODO(a.kazantsev): add check for transposed weights.
            for kernel in range(self.effective_shape[0]):
                for chan in range(self.effective_shape[1]):
                    self.mask[kernel, chan] = not (
                        kernel % self.grouping == chan % self.grouping)
        else:
            assert self.mask.shape == self.effective_shape

        for vec in self.mask, self.weights:
            vec.initialize(device)

    def _gpu_init(self):
        self.build_program(cache_file_name="zero_filling_%d" % self.grouping,
                           dtype=self.weights.dtype)

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
