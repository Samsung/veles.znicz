"""
Created on May 30, 2014

Activation functions (:class:`ActivationForward`) and their coupled GD units
(:class:`ActivationBackward`).

"""

from __future__ import division
import numpy
from zope.interface import implementer

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit
import veles.error as error
from veles.memory import eq_addr, ravel
from veles.znicz.nn_units import Forward, GradientDescentBase


class Activation(AcceleratedUnit):
    hide_from_registry = True

    def init_unpickled(self):
        super(Activation, self).init_unpickled()
        self.sources_["activation"] = {}


@implementer(IOpenCLUnit, ICUDAUnit)
class ActivationForward(Forward, Activation):
    MAPPING = set()

    def initialize(self, device, **kwargs):
        super(ActivationForward, self).initialize(device, **kwargs)

        if not self.output:
            self.output.reset(numpy.zeros_like(self.input.mem))
        else:
            assert self.output.shape == self.input.shape

        self.init_vectors(self.input, self.output)

    def _gpu_init(self):
        dtype = self.input.dtype
        self.build_program(
            {"OUTPUT_SIZE": self.input.size},
            "%s_%d" % (self.__class__.__name__, self.input.size),
            dtype=dtype)
        self.assign_kernel(self.kernel_name)
        self._set_activation_args()

    def ocl_init(self):
        self._gpu_init()
        self._global_size = (self.input.size,)
        self._local_size = None

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (
            int(numpy.ceil(self.input.size / block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def _set_activation_args(self):
        self.set_args(self.input, self.output)

    def cpu_prerun(self, make_raveled, copy_in2out):
        if make_raveled:
            inp = ravel(self.input.mem)
            out = ravel(self.output.mem)
        else:
            inp = self.input.mem
            out = self.output.mem
        if eq_addr(inp, out):
            self.output.map_write()
        else:
            self.output.map_invalidate()
            self.input.map_read()
            if copy_in2out:
                numpy.copyto(out, inp)
        return inp, out

    def _gpu_run(self):
        self.unmap_vectors(self.input, self.output)
        self.execute_kernel(self._global_size, self._local_size)

    def ocl_run(self):
        self._gpu_run()

    def cuda_run(self):
        self._gpu_run()


@implementer(IOpenCLUnit, ICUDAUnit)
class ActivationBackward(GradientDescentBase, Activation):
    """Backward activation pass: err_input = err_output * F'(output).

    Attributes:
        err_input: backprogated error to compute (OUT).
        err_output: error for backpropagation (IN).
        output: output of the layer AFTER applying activation function (IN).
        input: output of the layer BEFORE applying activation function (IN).
    """
    MAPPING = set()

    def __init__(self, workflow, **kwargs):
        super(ActivationBackward, self).__init__(workflow, **kwargs)
        self.demand("output")

    def initialize(self, device, **kwargs):
        super(ActivationBackward, self).initialize(device=device, **kwargs)

        if not self.err_input:
            self.err_input.reset(numpy.zeros_like(self.err_output.mem))
        else:
            assert self.err_input.shape == self.err_output.shape

        if self.input:
            self.input.initialize(self.device)
        self.init_vectors(self.err_output, self.err_input)

    def _gpu_init(self):
        dtype = self.err_output.dtype
        self.build_program(
            {"OUTPUT_SIZE": self.err_output.size},
            "%s_%d" % (self.__class__.__name__, self.err_output.size),
            dtype=dtype)
        self.assign_kernel(self.kernel_name)
        self._set_activation_args()

    def ocl_init(self):
        self._gpu_init()
        self._global_size = (self.err_output.size,)
        self._local_size = None

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (
            int(numpy.ceil(self.err_output.size / block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def _set_activation_args(self):
        self.set_args(self.input, self.output, self.err_output, self.err_input)

    def cpu_prerun(self, is_raveled, io_usage):
        inp = None
        out = None
        if is_raveled:
            if io_usage[0]:
                inp = ravel(self.input.mem)
            if io_usage[1]:
                out = ravel(self.output.mem)
            err_input = ravel(self.err_input.mem)
            err_output = ravel(self.err_output.mem)
        else:
            if io_usage[0]:
                inp = self.input.mem
            if io_usage[1]:
                out = self.output.mem
            err_input = self.err_input.mem
            err_output = self.err_output.mem
        if eq_addr(err_input, err_output):
            self.err_input.map_write()
        else:
            self.err_input.map_invalidate()
            self.err_output.map_read()
        if io_usage[0]:
            self.input.map_read()
        if io_usage[1]:
            self.output.map_read()
        return inp, out, err_input, err_output

    def _gpu_run(self):
        self.unmap_vectors(self.output, self.err_input, self.err_output)
        self.execute_kernel(self._global_size, self._local_size)

    def ocl_run(self):
        self._gpu_run()

    def cuda_run(self):
        self._gpu_run()


class ForwardTanh(ActivationForward):
    """Forward pass for y = 1.7159 * tanh(0.6666 * x).
    """

    kernel_name = "forward_tanh"
    MAPPING = {"activation_tanh"}

    def cpu_run(self):
        _, out = self.cpu_prerun(make_raveled=False, copy_in2out=True)
        numpy.tanh(out, out)
        out *= 1.7159


class BackwardTanh(ActivationBackward):
    """Backward pass for :class:`ForwardTanh`.
    """

    kernel_name = "backward_tanh"
    MAPPING = {"activation_tanh"}

    def cpu_run(self):
        _, output, err_input, err_output = \
            self.cpu_prerun(is_raveled=False, io_usage=(False, True))
        numpy.multiply(
            err_output, output * output * (-0.388484177) + 1.14381894,
            err_input)


class ForwardSigmoid(ActivationForward):
    """Forward pass for y = 1.0 / (1.0 + exp(-x)).
    """

    kernel_name = "forward_sigmoid"
    MAPPING = {"activation_sigmoid"}

    def cpu_run(self):
        _, out = self.cpu_prerun(make_raveled=False, copy_in2out=True)
        numpy.reciprocal(1.0 + numpy.exp(-out), out)


class BackwardSigmoid(ActivationBackward):
    """Backward pass for :class:`ForwardSigmoid`.
    """

    kernel_name = "backward_sigmoid"
    MAPPING = {"activation_sigmoid"}

    def cpu_run(self):
        _, output, err_input, err_output = \
            self.cpu_prerun(is_raveled=False, io_usage=(False, True))
        numpy.multiply(err_output, output * (1.0 - output), err_input)


class ForwardMul(ActivationForward):
    """Forward pass for :math:`y = k x`.
    """

    kernel_name = "forward_mul"
    MAPPING = {"activation_mul"}

    def __init__(self, workflow, **kwargs):
        super(ForwardMul, self).__init__(workflow, **kwargs)
        self._factor = kwargs.get("factor")

    def init_unpickled(self):
        super(ForwardMul, self).init_unpickled()
        self._cl_const = None

    def generate_data_for_slave(self, slave):
        return self.factor

    def apply_data_from_master(self, data):
        if self.factor != data:
            self.info("Setting factor to %.6f", data)
            self.factor = data

    def generate_data_for_master(self):
        return self.factor

    def apply_data_from_slave(self, data, slave):
        if data is None:
            return
        if self.factor is None:
            self.factor = data
        else:
            self.factor = min(self.factor, data)

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, value):
        self._factor = None if value is None else float(value)
        if self._kernel_ is None or value is None:
            return
        if self._cl_const is None:
            self._cl_const = numpy.ones(1, dtype=self.output.dtype)
        self._cl_const[0] = self._factor
        self.set_arg(2, self._cl_const)

    def ocl_init(self):
        super(ForwardMul, self).ocl_init()
        self.factor = self._factor

    def run(self):
        if self.factor is None:  # autoset factor from first minibatch
            self.input.map_read()
            mx = numpy.fabs(self.input.mem).max()
            factor = 0.75 / mx if mx else 0.75
            self.info("Autosetting factor to %f", factor)
            self.factor = factor
        super(ForwardMul, self).run()

    def cpu_run(self):
        _, out = self.cpu_prerun(make_raveled=False, copy_in2out=True)
        out *= self.factor


class BackwardMul(ActivationBackward):
    """Backward pass for :class:`ForwardMul`.
    """

    kernel_name = "backward_mul"
    MAPPING = {"activation_mul"}

    def __init__(self, workflow, **kwargs):
        super(BackwardMul, self).__init__(workflow, **kwargs)
        self._factor = float(kwargs.get("factor", 1.0))

    def init_unpickled(self):
        super(BackwardMul, self).init_unpickled()
        self._cl_const = None

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, value):
        self._factor = float(value)
        if self._kernel_ is None:
            return
        if self._cl_const is None:
            self._cl_const = numpy.ones(1, dtype=self.output.dtype)
        self._cl_const[0] = self._factor
        self.set_arg(4, self._cl_const)

    def ocl_init(self):
        super(BackwardMul, self).ocl_init()
        self.factor = self._factor

    def cpu_run(self):
        _, _, err_input, err_output = \
            self.cpu_prerun(is_raveled=False, io_usage=(False, False))
        err_input[:] = err_output[:] * self.factor


class ForwardRELU(ActivationForward):
    """
    This activation is taken from article
        *"ImageNet Classification with Deep Convolutional Neural Networks" \
        (sec 3.1)*.

    Forward pass:
        :math:`y = \\log(1 + \\exp(x).`
    """

    kernel_name = "forward_relu"
    MAPPING = {"activation_relu"}

    def cpu_run(self):
        inp, out = self.cpu_prerun(make_raveled=False, copy_in2out=False)
        out[:] = numpy.where(inp > 15, inp, numpy.log(numpy.exp(inp) + 1.0))


class BackwardRELU(ActivationBackward):
    """Backward pass for :class:`ForwardRELU`
    """

    kernel_name = "backward_relu"
    MAPPING = {"activation_relu"}

    def cpu_run(self):
        _, output, err_input, err_output = \
            self.cpu_prerun(is_raveled=False, io_usage=(False, True))
        numpy.multiply(err_output, 1.0 - numpy.exp(-output), err_input)


class ForwardStrictRELU(ActivationForward):
    """
    Forward pass for :math:`y = \\max(0, x)`.
    """

    kernel_name = "forward_strict_relu"
    MAPPING = {"activation_str"}

    def cpu_run(self):
        inp, out = self.cpu_prerun(make_raveled=False, copy_in2out=False)
        out[...] = numpy.where(numpy.greater(inp, 0), out, 0)

    # IDistributable implementation
    def generate_data_for_slave(self, slave):
        return None

    def generate_data_for_master(self):
        return None

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        pass

    def drop_slave(self, slave):
        pass


class BackwardStrictRELU(ActivationBackward):
    """
    Backward pass for :class:`ForwardStrictRELU`.

    :math:`x = \\max(y, 0)`
    """

    kernel_name = "backward_strict_relu"
    MAPPING = {"activation_str"}

    def cpu_run(self):
        _, output, err_input, err_output = \
            self.cpu_prerun(is_raveled=False, io_usage=(False, True))
        numpy.multiply(err_output, numpy.greater(output, 0), err_input)

    # IDistributable implementation
    def generate_data_for_slave(self, slave):
        return None

    def generate_data_for_master(self):
        return None

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        pass

    def drop_slave(self, slave):
        pass


class ForwardLog(ActivationForward):
    """Forward pass for :math:`y = \\log(x + \\sqrt{x^2 + 1})`.
    """

    MAPPING = {"activation_log"}

    def initialize(self, device, **kwargs):
        if (self.output is self.input or
            (self.output is not None and self.output.mem is not None and
             eq_addr(self.output.mem, self.input.mem))):
            raise error.BadFormatError("in_place for this unit is prohibited")
        super(ForwardLog, self).initialize(device=device, **kwargs)

    def ocl_init(self):
        self.assign_kernel("forward_log")
        self._set_activation_args()

    def cpu_run(self):
        inp, out = self.cpu_prerun(make_raveled=False, copy_in2out=False)
        numpy.log(inp + numpy.sqrt(numpy.square(inp) + 1), out)


class BackwardLog(ActivationBackward):
    """Backward pass for :class:`ForwardLog`.
    """

    MAPPING = {"activation_log"}

    def initialize(self, device, **kwargs):
        if (self.input is None or self.input.mem is None or
            (self.output is not None and
             eq_addr(self.input.mem, self.output.mem))):
            raise error.BadFormatError(
                "input should be set and should not be equal to output")
        super(BackwardLog, self).initialize(device=device, **kwargs)

    def ocl_init(self):
        self.assign_kernel("backward_log")
        self._set_activation_args()

    def cpu_run(self):
        inp, _, err_input, err_output = \
            self.cpu_prerun(is_raveled=False, io_usage=(True, False))
        numpy.multiply(
            err_output, numpy.reciprocal(numpy.sqrt(numpy.square(inp) + 1)),
            err_input)


class ForwardTanhLog(ActivationForward):
    """Forward pass for hybrid tanh-log function.
    """
    d = 3
    a = 0.242528761112
    b = 305.459953195
    kernel_name = "forward_tanhlog"
    MAPPING = {"activation_tanhlog"}

    def initialize(self, device, **kwargs):
        if (id(self.output) == id(self.input) or
            (self.output is not None and self.output.mem is not None and
             eq_addr(self.output.mem, self.input.mem))):
            raise error.BadFormatError("in_place for this unit is prohibited")
        super(ForwardTanhLog, self).initialize(device=device, **kwargs)

    def cpu_run(self):
        inp, out = self.cpu_prerun(make_raveled=True, copy_in2out=False)
        for i, x in enumerate(inp):
            if x > ForwardTanhLog.d:
                y = numpy.log(x * ForwardTanhLog.b) * ForwardTanhLog.a
            elif x < -ForwardTanhLog.d:
                y = numpy.log(x * (-ForwardTanhLog.b)) * (-ForwardTanhLog.a)
            else:
                y = 1.7159 * numpy.tanh(x * 0.6666)
            out[i] = y


class BackwardTanhLog(ActivationBackward):
    """Backward pass for hybrid tanh-log function.
    """

    kernel_name = "backward_tanhlog"
    MAPPING = {"activation_tanhlog"}

    def __init__(self, workflow, **kwargs):
        super(BackwardTanhLog, self).__init__(workflow, **kwargs)
        self.demand("output")

    def initialize(self, device, **kwargs):
        if (not self.input or
            (self.output is not None and
             eq_addr(self.input.mem, self.output.mem))):
            raise error.BadFormatError(
                "input should be set and should not be equal to output")
        super(BackwardTanhLog, self).initialize(device=device, **kwargs)
        self.output.initialize(self.device)

    def cpu_run(self):
        inp, out, err_input, err_output = \
            self.cpu_prerun(is_raveled=True, io_usage=(True, True))
        for i, x in enumerate(inp):
            if x > ForwardTanhLog.d:
                y = ForwardTanhLog.a / x
            elif x < -ForwardTanhLog.d:
                y = -ForwardTanhLog.a / x
            else:
                y = numpy.square(out[i]) * (-0.388484177) + 1.14381894
            err_input[i] = err_output[i] * y

    def _set_activation_args(self):
        self.set_args(self.input, self.output, self.err_output, self.err_input)


class ForwardSinCos(ActivationForward):
    """Forward pass for y = sin(x) if idx(x) is odd else cos(x).
    """

    kernel_name = "forward_sincos"
    MAPPING = {"activation_sincos"}

    def initialize(self, device, **kwargs):
        if (id(self.output) == id(self.input) or
            (self.output is not None and self.output.mem is not None and
             eq_addr(self.output.mem, self.input.mem))):
            raise error.BadFormatError("in_place for this unit is prohibited")
        super(ForwardSinCos, self).initialize(device=device, **kwargs)

    def cpu_run(self):
        inp, out = self.cpu_prerun(make_raveled=True, copy_in2out=False)
        out[1::2] = numpy.sin(inp[1::2])
        out[0::2] = numpy.cos(inp[0::2])


class BackwardSinCos(ActivationBackward):
    """Backward pass for :class:`ForwardSinCos`.
    """

    kernel_name = "backward_sincos"
    MAPPING = {"activation_sincos"}

    def initialize(self, device, **kwargs):
        if not self.input:
            raise error.BadFormatError(
                "input should be set and should not be equal to output")
        super(BackwardSinCos, self).initialize(device=device, **kwargs)

    def cpu_run(self):
        inp, _, err_input, err_output = \
            self.cpu_prerun(is_raveled=True, io_usage=(True, False))
        err_input[1::2] = err_output[1::2] * numpy.cos(inp[1::2])
        err_input[0::2] = err_output[0::2] * (-numpy.sin(inp[0::2]))
