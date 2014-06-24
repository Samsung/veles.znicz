"""
Created on May 30, 2014

An activation layers.
"""


import numpy
from zope.interface import implementer

from veles import formats, OpenCLUnit
from veles.opencl_units import IOpenCLUnit
from veles.znicz.nn_units import Forward, GradientDescentBase
from veles import error


class Activation(OpenCLUnit):
    def init_unpickled(self):
        super(Activation, self).init_unpickled()
        self.cl_sources_["activation.cl"] = {}


@implementer(IOpenCLUnit)
class ActivationForward(Forward, Activation):
    def initialize(self, device, **kwargs):
        super(ActivationForward, self).initialize(device, **kwargs)

        dtype = self.input.mem.dtype

        if (self.output.mem is None or
                self.output.mem.size != self.input.mem.size):
            self.output.reset()
            self.output.mem = numpy.zeros_like(self.input.mem)

        self.input.initialize(device)
        self.output.initialize(device)

        if device is None:
            return

        self.build_program({}, "%s.cl" % self.__class__.__name__,
                           dtype=dtype)

    def set_args(self):
        super(ActivationForward, self).set_args(
            self.input, self.output)

    def ocl_run(self):
        self.input.unmap()
        self.output.unmap()
        self.execute_kernel((self.input.mem.size,), None)


@implementer(IOpenCLUnit)
class ActivationBackward(GradientDescentBase, Activation):
    """Backward activation pass: err_input = err_output * F'(output).

    Attributes:
        err_input: backprogated error to compute (OUT).
        err_output: error for backpropagation (IN).
        output: output of the layer AFTER applying activation function (IN).
        input: output of the layer BEFORE applying activation function (IN).
    """
    def initialize(self, device, **kwargs):
        super(ActivationBackward, self).initialize(device=device, **kwargs)

        dtype = self.err_output.mem.dtype

        if (self.err_input.mem is None or
                self.err_input.mem.size != self.err_output.mem.size):
            self.err_input.reset()
            self.err_input.mem = numpy.zeros_like(self.err_output.mem)

        if self.input is not None:
            self.input.initialize(device)
        self.output.initialize(device)
        self.err_output.initialize(device)
        self.err_input.initialize(device)

        if device is None:
            return

        self.build_program({}, "%s.cl" % self.__class__.__name__,
                           dtype=dtype)

    def set_args(self):
        super(ActivationBackward, self).set_args(
            self.input, self.output, self.err_output, self.err_input)

    def ocl_run(self):
        self.err_output.unmap()
        self.err_input.unmap()
        self.execute_kernel((self.err_output.mem.size,), None)


@implementer(IOpenCLUnit)
class ForwardTanh(ActivationForward):
    """Forward pass for y = 1.7159 * tanh(0.6666 * x).
    """
    def initialize(self, device, **kwargs):
        super(ForwardTanh, self).initialize(device=device, **kwargs)
        if device is None:
            return
        self.assign_kernel("forward_tanh")
        self.set_args()

    def cpu_run(self):
        inp = self.input.mem
        out = self.output.mem
        if formats.eq_addr(inp, out):
            self.output.map_write()
        else:
            self.output.map_invalidate()
            self.input.map_read()
            numpy.copyto(out, inp)
        out *= 0.6666
        numpy.tanh(out, out)
        out *= 1.7159


@implementer(IOpenCLUnit)
class BackwardTanh(ActivationBackward):
    """Backward pass for y = 1.7159 * tanh(0.6666 * x).
    """
    def initialize(self, device, **kwargs):
        super(BackwardTanh, self).initialize(device=device, **kwargs)
        if device is None:
            return
        self.assign_kernel("backward_tanh")
        self.set_args()

    def cpu_run(self):
        err_input = self.err_input.mem
        err_output = self.err_output.mem
        output = self.output.mem
        if formats.eq_addr(err_input, err_output):
            self.err_input.map_write()
        else:
            self.err_input.map_invalidate()
            self.err_output.map_read()
        self.output.map_read()
        numpy.multiply(err_output,
                       output * output * (-0.388484177) + 1.14381894,
                       err_input)


@implementer(IOpenCLUnit)
class ForwardStrictRELU(ActivationForward):
    """Forward pass for y = max(0, x).
    """
    def initialize(self, device, **kwargs):
        super(ForwardStrictRELU, self).initialize(device=device, **kwargs)
        if device is None:
            return
        self.assign_kernel("forward_strict_relu")
        self.set_args()

    def cpu_run(self):
        inp = self.input.mem
        out = self.output.mem
        if formats.eq_addr(inp, out):
            self.output.map_write()
        else:
            self.output.map_invalidate()
            self.input.map_read()
        out[...] = numpy.where(numpy.greater(inp, 0), inp, 0)


@implementer(IOpenCLUnit)
class BackwardStrictRELU(ActivationBackward):
    """Backward pass for y = max(0, x).
    """
    def initialize(self, device, **kwargs):
        super(BackwardStrictRELU, self).initialize(device=device, **kwargs)
        if device is None:
            return
        self.assign_kernel("backward_strict_relu")
        self.set_args()

    def cpu_run(self):
        err_input = self.err_input.mem
        err_output = self.err_output.mem
        output = self.output.mem
        if formats.eq_addr(err_input, err_output):
            self.err_input.map_write()
        else:
            self.err_input.map_invalidate()
            self.err_output.map_read()
        self.output.map_read()
        numpy.multiply(err_output, numpy.greater(output, 0), err_input)


@implementer(IOpenCLUnit)
class ForwardLog(ActivationForward):
    """Forward pass for y = log(x + sqrt(x * x + 1)).
    """
    def initialize(self, device, **kwargs):
        if (id(self.output) == id(self.input) or
            (self.output is not None and self.output.mem is not None and
             formats.eq_addr(self.output.mem, self.input.mem))):
            raise error.BadFormatError("in_place for this unit is prohibited")
        super(ForwardLog, self).initialize(device=device, **kwargs)
        if device is None:
            return
        self.assign_kernel("forward_log")
        self.set_args()

    def cpu_run(self):
        inp = self.input.mem
        out = self.output.mem
        if formats.eq_addr(inp, out):
            self.output.map_write()
        else:
            self.output.map_invalidate()
            self.input.map_read()
        numpy.log(inp + numpy.sqrt(numpy.square(inp) + 1), out)


@implementer(IOpenCLUnit)
class BackwardLog(ActivationBackward):
    """Backward pass for y = log(x + sqrt(x * x + 1)).
    """
    def initialize(self, device, **kwargs):
        if (self.input is None or self.input.mem is None or
            (self.output is not None and
             formats.eq_addr(self.input.mem, self.output.mem))):
            raise error.BadFormatError(
                "input should be set and should not be equal to output")
        super(BackwardLog, self).initialize(device=device, **kwargs)
        if device is None:
            return
        self.assign_kernel("backward_log")
        self.set_args()

    def cpu_run(self):
        inp = self.input.mem
        err_input = self.err_input.mem
        err_output = self.err_output.mem
        if formats.eq_addr(err_input, err_output):
            self.err_input.map_write()
        else:
            self.err_input.map_invalidate()
            self.err_output.map_read()
        self.input.map_read()
        numpy.multiply(
            err_output, numpy.reciprocal(numpy.sqrt(numpy.square(inp) + 1)),
            err_input)


@implementer(IOpenCLUnit)
class ForwardSinCos(ActivationForward):
    """Forward pass for y = sin(x) if idx(x) is odd else cos(x).
    """
    def initialize(self, device, **kwargs):
        if (id(self.output) == id(self.input) or
            (self.output is not None and self.output.mem is not None and
             formats.eq_addr(self.output.mem, self.input.mem))):
            raise error.BadFormatError("in_place for this unit is prohibited")
        super(ForwardSinCos, self).initialize(device=device, **kwargs)
        if device is None:
            return
        self.assign_kernel("forward_sincos")
        self.set_args()

    def cpu_run(self):
        inp = formats.ravel(self.input.mem)
        out = formats.ravel(self.output.mem)
        if formats.eq_addr(inp, out):
            self.output.map_write()
        else:
            self.output.map_invalidate()
            self.input.map_read()
        out[1::2] = numpy.sin(inp[1::2])
        out[0::2] = numpy.cos(inp[0::2])


@implementer(IOpenCLUnit)
class BackwardSinCos(ActivationBackward):
    """Backward pass for y = sin(x) if idx(x) is odd else cos(x).
    """
    def initialize(self, device, **kwargs):
        if (self.input is None or self.input.mem is None or
            (self.output is not None and
             formats.eq_addr(self.input.mem, self.output.mem))):
            raise error.BadFormatError(
                "input should be set and should not be equal to output")
        super(BackwardSinCos, self).initialize(device=device, **kwargs)
        if device is None:
            return
        self.assign_kernel("backward_sincos")
        self.set_args()

    def cpu_run(self):
        inp = formats.ravel(self.input.mem)
        err_input = formats.ravel(self.err_input.mem)
        err_output = formats.ravel(self.err_output.mem)
        if formats.eq_addr(err_input, err_output):
            self.err_input.map_write()
        else:
            self.err_input.map_invalidate()
            self.err_output.map_read()
        self.input.map_read()
        err_input[1::2] = err_output[1::2] * numpy.cos(inp[1::2])
        err_input[0::2] = err_output[0::2] * (-numpy.sin(inp[0::2]))
