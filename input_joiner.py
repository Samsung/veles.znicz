"""
Created on Oct 29, 2013

Joins several inpus into one continuous output.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division
import numpy
import opencl4py as cl
from zope.interface import implementer

import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.opencl_units import OpenCLUnit, IOpenCLUnit


@implementer(IOpenCLUnit)
class InputJoiner(OpenCLUnit):
    """Joins several minibatch inputs into one continuous minibatch output.

    Must be assigned before initialize():
        inputs

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        inputs: list of inputs of type formats.Vector().
        output: formats.Vector().
        output_sample_shape: shape of an output sample, if None,
                             will be a plain sum of input sample shapes.
        minibatch_size: size of the minibatch (will be set to the minimum
                        of the first shapes from the inputs
                        if not provided prior to the initialize)
    """
    def __init__(self, workflow, **kwargs):
        output_sample_shape = kwargs.get("output_sample_shape")
        inputs = kwargs.get("inputs")
        kwargs["output_sample_shape"] = output_sample_shape
        kwargs["inputs"] = inputs
        super(InputJoiner, self).__init__(workflow, **kwargs)
        self.inputs = [] if inputs is None else inputs
        self.output = formats.Vector()
        self.output_sample_shape = output_sample_shape
        self.cl_const = numpy.zeros(4, dtype=numpy.int32)
        self.minibatch_size = [None]

    def init_unpickled(self):
        super(InputJoiner, self).init_unpickled()
        self.cl_sources_["join.cl"] = {}

    def initialize(self, device, **kwargs):
        if not len(self.inputs):
            raise error.BadFormatError("inputs should not be empty")

        super(InputJoiner, self).initialize(device=device, **kwargs)

        if self.minibatch_size[0] is None:
            minibatch_size = self.inputs[0].mem.shape[0]
            for i in range(1, len(self.inputs)):
                minibatch_size = min(minibatch_size,
                                     self.inputs[i].mem.shape[0])
            self.minibatch_size[0] = minibatch_size
        else:
            minibatch_size = self.minibatch_size[0]

        if self.output_sample_shape is None:
            self.output_sample_shape = [0]
            for inp in self.inputs:
                if inp.mem is None:
                    raise error.BadFormatError(
                        "output_sample_shape should be provided "
                        "if any of the inputs was not initialized "
                        "before this point")
                self.output_sample_shape[0] += inp.mem.size // inp.mem.shape[0]

        sh = [minibatch_size]
        sh.extend(self.output_sample_shape)
        if self.output.mem is None or self.output.mem.size != numpy.prod(sh):
            self.output.reset()
            self.output.mem = numpy.zeros(sh, dtype=self.inputs[0].mem.dtype)
        else:
            self.output.mem = formats.reshape(self.output.mem, sh)

        self.output.initialize(self)

        if self.device is not None:
            InputJoiner.ocl_init(self, device)

    def ocl_init(self, device):
        defines = {
            'etype':
            opencl_types.numpy_dtype_to_opencl(self.output.mem.dtype)
        }
        self.build_program(
            defines, "join_%s.cl" %
            "_".join(str(x) for x in self.output_sample_shape))
        self.assign_kernel("join2")
        self.set_args(self.output)

    def cpu_run(self):
        self.output.map_invalidate()  # we will update output on CPU
        minibatch_size = self.minibatch_size[0]
        low = 0
        output_sample_size = self.output.mem.size // self.output.mem.shape[0]
        for inp in self.inputs:
            inp.map_read()
            high = min(low + inp.mem.size // inp.mem.shape[0],
                       output_sample_size)
            if low >= high:
                break
            self.output.mem[:minibatch_size, low:high] = (
                inp[:minibatch_size, :high - low])
            low = high

    def ocl_run(self):
        self.output.unmap()  # we will update output on GPU
        minibatch_size = self.minibatch_size[0]
        low = 0
        output_sample_size = self.output.mem.size // self.output.mem.shape[0]
        self.cl_const[3] = output_sample_size
        self.set_arg(6, self.cl_const[3:4])
        a = None
        a_size = 0
        b = None
        for inp in self.inputs:
            inp.unmap()  # we will use input on GPU
            if a is None:
                a = inp
                a_size = a.mem.size // a.mem.shape[0]
                continue
            b = inp
            b_size = b.mem.size // b.mem.shape[0]
            high = min(low + a_size + b_size,
                       output_sample_size)
            if low >= high:
                break
            self.cl_const[0] = a_size
            self.cl_const[1] = b_size
            self.cl_const[2] = low
            self.set_args(cl.skip, a, b, self.cl_const[0:1],
                          self.cl_const[1:2], self.cl_const[2:3])
            global_size = [high - low, minibatch_size]
            self.execute_kernel(global_size, None)
            low = high
            a = None
            a_size = 0
            b = None
        if a is not None:
            b_size = (b.mem.size // b.mem.shape[0] if b is not None else 0)
            high = min(low + a_size + b_size,
                       output_sample_size)
            if low < high:
                self.cl_const[0] = a_size
                self.cl_const[1] = b_size
                self.cl_const[2] = low
                self.set_args(cl.skip, a, b, self.cl_const[0:1],
                              self.cl_const[1:2], self.cl_const[2:3])
                if b is None:
                    self.set_arg(2, None)
                global_size = [high - low, minibatch_size]
                self.execute_kernel(global_size, None)
