"""
Created on Apr 25, 2014

A dropout layer. It is a signal repeater with some repeating channels set to 0.
Inputs to be disabled are randomly selected each forward proparation.

Detailed description given in article by Krizhevsky, Sutskever and Hinton:
"ImageNet Classification with Deep Convolutional Neural Networks" (sec. 4.2).
"""

from __future__ import division
import numpy
import opencl4py as cl
from zope.interface import implementer

from veles import formats
import veles.prng as random_generator
from veles.opencl_units import OpenCLUnit, IOpenCLUnit
from veles.znicz.nn_units import Forward, GradientDescentBase
from veles.distributable import IDistributable, TriviallyDistributable


class Dropout(OpenCLUnit, TriviallyDistributable):
    """
    A base class for forward and backward units of local
    response normalization.
    """
    def __init__(self, workflow, **kwargs):
        super(Dropout, self).__init__(workflow, **kwargs)
        self.dropout_ratio = kwargs.get("dropout_ratio")

    def init_unpickled(self):
        super(Dropout, self).init_unpickled()
        self.cl_sources_["dropout"] = {}

    @property
    def dropout_ratio(self):
        """ Gets the relative amount of weights to disable.
        """
        return self._dropout_ratio

    @dropout_ratio.setter
    def dropout_ratio(self, value):
        """ Sets the relative amount of weights to disable.
        """
        assert value is None or 0. < value < 1.
        self._dropout_ratio = value


@implementer(IOpenCLUnit)
class DropoutForward(Forward, Dropout):
    """
    Forward propagation of dropout layer.
    """
    MIN_RANDOM_STATE = 0
    MAX_RANDOM_STATE = 0x100000000
    MAPPING = {"dropout"}

    def __init__(self, workflow, **kwargs):
        super(DropoutForward, self).__init__(workflow, **kwargs)
        self.mask = formats.Vector()  # dropout mask
        self.states = formats.Vector()
        self.rand = random_generator.get()

    @Dropout.dropout_ratio.setter
    def dropout_ratio(self, value):
        Dropout.dropout_ratio.fset(self, value)
        if hasattr(self, "input") and self.input is not None:
            self.calc_mask()

    def initialize(self, device, **kwargs):
        super(DropoutForward, self).initialize(device=device, **kwargs)
        self.mask.mem = numpy.empty_like(self.input.mem)
        self.states.mem = self.rand.randint(
            low=DropoutForward.MIN_RANDOM_STATE,
            high=DropoutForward.MAX_RANDOM_STATE,
            size=self.input.mem.size * 4).astype(numpy.uint32)
        if (self.output.mem is None or
                self.output.mem.size != self.input.mem.size):
            self.output.reset()
            self.output.mem = numpy.zeros_like(self.input.mem)

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.states.initialize(self.device)
        self.mask.initialize(self.device)

        self.backend_init()

    def ocl_init(self):
        self._threshold_arg_ = numpy.empty(1, dtype=numpy.uint64)
        self._pass_arg_ = numpy.empty(1, dtype=self.input.mem.dtype)

        self.build_program({}, "%s_%s" %
                           (self.__class__.__name__,
                            "x".join(str(x) for x in self.input.shape)),
                           dtype=self.input.mem.dtype)

        self.assign_kernel("dropout_forward")
        self.set_args(self.input, cl.skip, cl.skip, self.states, self.mask,
                      self.output)

    def calc_mask(self):
        leave_ratio = 1.0 - self.dropout_ratio
        self.rand.fill(self.mask.mem, -self.dropout_ratio, leave_ratio)
        numpy.maximum(self.mask.mem, 0, self.mask.mem)
        numpy.ceil(self.mask.mem, self.mask.mem)
        self.mask.mem[:] = (self.mask.mem.astype(self.input.mem.dtype) /
                            leave_ratio)

    def cpu_run(self):
        self.output.map_invalidate()
        self.input.map_read()
        if not self.forward_mode:
            self.mask.map_invalidate()
            self.calc_mask()
            numpy.multiply(self.input.mem.ravel(), self.mask.mem.ravel(),
                           formats.ravel(self.output.mem))
        else:
            self.output.mem[:] = self.input.mem

    def ocl_run(self):
        self.input.unmap()
        self.output.unmap()
        if not self.forward_mode:
            self.states.unmap()
            self.mask.unmap()
            self._threshold_arg_[0] = ((1 << 64) - 1.0) * self.dropout_ratio
            self._pass_arg_[0] = 1.0 / (1.0 - self.dropout_ratio)
            self.set_arg(1, self._threshold_arg_)
            self.set_arg(2, self._pass_arg_)
            self.execute_kernel((self.input.mem.size,), None)
        else:
            self.device.queue_.copy_buffer(
                self.input.devmem, self.output.devmem, 0, 0,
                self.output.nbytes, need_event=False)


@implementer(IOpenCLUnit, IDistributable)
class DropoutBackward(GradientDescentBase, Dropout):
    """
    Backward propagation of droupout layer.
    """

    MAPPING = {"dropout"}

    def __init__(self, workflow, **kwargs):
        self.mask = None  # dropout mask (should be given from forward unit)
        super(DropoutBackward, self).__init__(workflow, **kwargs)

    def initialize(self, device, **kwargs):
        super(DropoutBackward, self).initialize(device=device, **kwargs)

        if (self.err_input.mem is None or
                self.err_input.mem.size != self.err_output.mem.size):
            self.err_input.reset()
            self.err_input.mem = numpy.zeros_like(self.err_output.mem)

        if self.device is None:
            return  # if is master

        self.err_output.initialize(self.device)
        self.err_input.initialize(self.device)

        self.backend_init()

    def ocl_init(self):
        self.build_program({}, "%s_%s" %
                           (self.__class__.__name__,
                            "x".join(str(x) for x in self.err_input)),
                           dtype=self.err_output.mem.dtype)
        self.assign_kernel("dropout_backward")
        self.set_args(self.mask, self.err_output, self.err_input)

    def cpu_run(self):
        if formats.eq_addr(self.err_input.mem, self.err_output.mem):
            self.err_output.map_write()
        else:
            self.err_output.map_read()
            self.err_input.map_invalidate()
        self.mask.map_read()
        numpy.multiply(self.err_output.mem.ravel(), self.mask.mem.ravel(),
                       formats.ravel(self.err_input.mem))

    def ocl_run(self):
        self.err_output.unmap()
        self.err_input.unmap()
        self.mask.unmap()
        self.execute_kernel((self.err_output.mem.size,), None)
