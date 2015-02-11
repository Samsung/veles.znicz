"""
Created on Jul 10, 2014

Depoling unit.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import numpy
from zope.interface import implementer

import veles.error as error
from veles.accelerated_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
class Depooling(nn_units.Forward):
    """Depooling unit for *Max* poolings.

    Must be assigned before initialize():
        input
        output_offset
        output_shape_source

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: data to depool.
        output: depooled data.
        output_offset: input_offset from the corresponding pooling unit.
        output_shape_source: Vector to get output shape from.
        krn_output_clear_: kernel for zeroing the output.
    """
    def __init__(self, workflow, **kwargs):
        super(Depooling, self).__init__(workflow, **kwargs)
        self.output_offset = None
        self.demand("output_offset", "output_shape_source")

    def init_unpickled(self):
        super(Depooling, self).init_unpickled()
        self.sources_["depooling"] = {}
        self.krn_output_clear_ = None

    def initialize(self, device, **kwargs):
        super(Depooling, self).initialize(device, **kwargs)

        if self.output_offset.size != self.input.size:
            raise error.BadFormatError("output_offset.size != input.size")

        if self.output_offset.dtype != numpy.int32:
            raise error.BadFormatError("output_offset.dtype != numpy.int32")

        if not self.output:
            self.output.reset(numpy.zeros(self.output_shape_source.shape,
                                          dtype=self.input.dtype))
        else:
            assert self.output.shape == self.output_shape_source.shape

        self.init_vectors(self.input, self.output_offset, self.output)

    def _gpu_init(self):
        self.build_program(
            {"INPUT_SIZE": self.input.size}, "%s_%s" %
            (self.__class__.__name__,
             "_".join(str(i) for i in self.input.shape)),
            dtype=self.input.dtype)

        self.assign_kernel("feed_layer")
        self.set_args(self.input, self.output_offset, self.output)

    def ocl_init(self):
        self._gpu_init()

        if self.krn_output_clear_ is None:
            self.krn_output_clear_ = self.get_kernel("output_clear")
            self.krn_output_clear_.set_arg(0, self.output.devmem)

    def cuda_init(self):
        self._gpu_init()

        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (
            int(numpy.ceil(self.input.size / block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def ocl_run(self):
        """Do gradient descent.
        """
        self.unmap_vectors(self.input, self.output_offset, self.output)
        self.execute_kernel([self.output.size], None, self.krn_output_clear_)
        self.execute_kernel([self.input.size], None)

    def cuda_run(self):
        self.unmap_vectors(self.input, self.output_offset, self.output)
        self.output.devmem.memset32_async()
        self.execute_kernel(self._global_size, self._local_size)

    def cpu_run(self):
        raise RuntimeError("Not implemented")

    def generate_data_for_slave(self):
        pass

    def apply_data_from_slave(self):
        pass
