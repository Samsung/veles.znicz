"""
Created on Jul 10, 2014

Depoling unit.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import numpy
from zope.interface import implementer

import veles.error as error
from veles.opencl_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
class Depooling(nn_units.Forward):
    """Depooling unit for *Max* poolings.

    Must be assigned before initialize():
        input
        output_offset
        get_output_shape_from

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: data to depool.
        output: depooled data.
        output_offset: input_offset from the corresponding pooling unit.
        get_output_shape_from: Vector to get output shape from.
        krn_output_clear_: kernel for zeroing the output.
    """
    def __init__(self, workflow, **kwargs):
        super(Depooling, self).__init__(workflow, **kwargs)
        self.output_offset = None
        self.demand("output_offset", "get_output_shape_from")

    def init_unpickled(self):
        super(Depooling, self).init_unpickled()
        self.cl_sources_["depooling.cl"] = {}
        self.krn_output_clear_ = None

    def initialize(self, device, **kwargs):
        super(Depooling, self).initialize(device, **kwargs)

        if self.output_offset.size != self.input.size:
            raise error.BadFormatError("output_offset.size != input.size")

        if self.output_offset.dtype != numpy.int32:
            raise error.BadFormatError("output_offset.dtype != numpy.int32")

        if (self.output.mem is None or
                self.output.size != self.get_output_shape_from.size):
            self.output.reset()
            self.output.mem = numpy.zeros(self.get_output_shape_from.shape,
                                          dtype=self.input.dtype)

        self.input.initialize(self.device)
        self.output_offset.initialize(self.device)
        self.output.initialize(self.device)

        if self.device is None:
            return

        if self.program_ is None:
            self.build_program(
                {}, "depooling_%s.cl" %
                "_".join(str(i) for i in self.input.shape),
                dtype=self.input.dtype)

            self.assign_kernel("feed_layer")
            self._set_args(self.input, self.output_offset, self.output)

            if self.krn_output_clear_ is None:
                self.krn_output_clear_ = self.get_kernel("output_clear")
                self.krn_output_clear_.set_arg(0, self.output.devmem)

    def ocl_run(self):
        """Do gradient descent.
        """
        self.input.unmap()
        self.output_offset.unmap()
        self.output.unmap()

        self.execute_kernel([self.output.size], None, self.krn_output_clear_)
        self.execute_kernel([self.input.size], None)

    def cpu_run(self):
        raise RuntimeError("Not implemented")

    def generate_data_for_slave(self):
        pass

    def apply_data_from_slave(self):
        pass
