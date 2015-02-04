import numpy
from zope.interface import implementer

from veles.accelerated_units import IOpenCLUnit
from veles.znicz.nn_units import Forward


@implementer(IOpenCLUnit)
class ChannelSplitter(Forward):

    def __init__(self, workflow, **kwargs):
        super(ChannelSplitter, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(ChannelSplitter, self).init_unpickled()
        self.cl_sources_["channel_splitting"] = {}

    def initialize(self, device, **kwargs):
        super(ChannelSplitter, self).initialize(device=device, **kwargs)
        output_shape = self.input.shape[0::3] + self.input.shape[1:3]
        if not self.output.mem:
            self.output.reset(numpy.zeros(output_shape, self.input.mem.dtype))
        else:
            assert self.output.shape == output_shape

        self.init_vectors(self.input, self.output)

    def ocl_init(self):
        self.build_program({}, "%s_%s" %
                           (self.__class__.__name__,
                            "x".join(str(x) for x in self.input.shape)),
                           dtype=self.input.dtype)

        self.assign_kernel("split_channels")
        self.set_args(
            self.input, self.output,
            numpy.array([self.input.shape[0]], dtype=numpy.int32),  # n_pics
            numpy.array([self.input.shape[3]], dtype=numpy.int32),  # n_chans
            numpy.array([self.input.shape[1]], dtype=numpy.int32),  # pic_h
            numpy.array([self.input.shape[2]], dtype=numpy.int32))  # pic_w

        self._global_size = [self.input.shape[1]]

    def cpu_run(self):
        self.output.map_invalidate()
        self.input.map_read()
        self.output.mem[:] = self.input.mem.swapaxes(3, 2).swapaxes(2, 1)[:]

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()
        self.input.unmap()
        self.execute_kernel(self._global_size, None)


@implementer(IOpenCLUnit)
class ChannelMerger(Forward):
    def __init__(self, workflow, **kwargs):
        super(ChannelMerger, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(ChannelMerger, self).init_unpickled()
        self.cl_sources_["channel_splitting"] = {}

    def initialize(self, device, **kwargs):
        super(ChannelMerger, self).initialize(device=device, **kwargs)
        output_shape = self.input.shape[0::2] + self.input.shape[3::-2]
        if not self.output:
            self.output.reset(numpy.zeros(output_shape, self.input.mem.dtype))
        else:
            assert self.output.shape == output_shape

        self.init_vectors(self.input, self.output)

    def ocl_init(self):
        self.build_program({}, "%s_%s" %
                           (self.__class__.__name__,
                            "x".join(str(x) for x in self.input.shape)),
                           dtype=self.input.dtype)

        self.assign_kernel("merge_channels")
        self.set_args(
            self.input, self.output,
            numpy.array([self.input.shape[0]], dtype=numpy.int32),  # n_pics
            numpy.array([self.input.shape[1]], dtype=numpy.int32),  # n_chans
            numpy.array([self.input.shape[2]], dtype=numpy.int32),  # pic_h
            numpy.array([self.input.shape[3]], dtype=numpy.int32))  # pic_w

        self._global_size = [self.input.shape[2]]

    def cpu_run(self):
        self.output.map_invalidate()
        self.input.map_read()
        self.output.mem[:] = self.input.mem.swapaxes(1, 2).swapaxes(2, 3)[:]

    def ocl_run(self):
        self.output.unmap()
        self.input.unmap()
        self.execute_kernel(self._global_size, None)
