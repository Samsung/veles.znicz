# encoding: utf-8
"""
Created on May 16, 2014

Dynamic adjust of learning rates of GD units. Learning rate are changed
    according to iteration number, each iteration.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
from __future__ import division

from math import floor
import six
from scipy.interpolate import interp1d
from zope.interface import implementer, Interface

from veles.units import IUnit, Unit
from veles.znicz.nn_units import GradientDescentBase
from veles.distributable import IDistributable
from veles.unit_registry import MappedUnitRegistry
from veles.verified import Verified


class LRAdjustPolicyRegistry(MappedUnitRegistry):
    mapping = "lradjustpolicy"
    base = object


@implementer(IUnit, IDistributable)
class LearningRateAdjust(Unit):
    """
    This unit should be linked from Decision to run with each minibatch.
    """

    def __init__(self, workflow, **kwargs):
        super(LearningRateAdjust, self).__init__(workflow, **kwargs)
        self._gd_units = []
        self._minibatches_count = 0
        self.lr_policy_name = kwargs.get("lr_policy_name", None)
        self.bias_lr_policy_name = kwargs.get("bias_lr_policy_name", None)
        self.lr_parameters = kwargs.get("lr_parameters", {})
        self.bias_lr_parameters = kwargs.get("bias_lr_parameters", {})
        self.notified = False

    def add_gd_unit(self, gd_unit):
        """
        Gradient unit should have learning_rate property.

        Args:
            gd_unit(:class:`GradientDescentBase`): gradient unit with
                    ``learning_rate`` parameter to manipulate.
            lr_policy(:class:`ILRPolicy`): callable object that takes `int`
                iteration number and returns :class:`float` **weight**
                learning rate
            bias_lr_function(:class:`ILRPolicy`): callable object that takes
                `int` iteration number and returns :class:`float` **bias**
                learning rate. if nothing is set - `lr_policy` is taken)
        """
        assert isinstance(gd_unit, GradientDescentBase)
        self.gate_skip = gd_unit.gate_skip
        self._gd_units.append(gd_unit)

    def adjust_learning_rate(
            self, lr_to_adjust, lr_policy_name, lr_parameters):

        if lr_policy_name is not None:
            lr = float(LRAdjustPolicyRegistry.lradjustpolicy[
                lr_policy_name](lr_to_adjust, **lr_parameters)(
                self._minibatches_count))
            if lr_to_adjust != lr:
                if not self.notified:
                    self.notified = True
            return lr
        else:
            return None

    def initialize(self, **kwargs):
        pass

    def run(self):
        """
        Adjusts learning rates of GD units according to ``lr_policy``
        Should be run every minibatch before GD units.
        """
        if self.is_slave:
            return

        self.notified = False

        for gd_unit in self._gd_units:
            lr = self.adjust_learning_rate(
                gd_unit.learning_rate, self.lr_policy_name, self.lr_parameters)
            if lr is not None:
                gd_unit.learning_rate = lr
            lr_bias = self.adjust_learning_rate(
                gd_unit.learning_rate_bias, self.bias_lr_policy_name,
                self.bias_lr_parameters)
            if lr_bias is not None:
                gd_unit.learning_rate_bias = lr_bias

        self._minibatches_count += 1

    # IDistributable implementation
    def generate_data_for_slave(self, slave):
        return None

    def generate_data_for_master(self):
        return True

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        if not bool(self.gate_block) and not bool(self.gate_skip):
            self.run()

    def drop_slave(self, slave):
        pass


# LEARNING RATE POLICIES:

class ILRPolicy(Interface):
    """
    An ILRPolicy must be a pickleable callable object,
        taking iteration number and returning actial learning rate.

    """
    def __call__(self, iter):
        """
        Attrs:
            iter(int): current iteration
        Returns:
            float: learning rate
        """


@six.add_metaclass(LRAdjustPolicyRegistry)
class PolicyBase(Verified):
    pass


@implementer(ILRPolicy)
class ExpPolicy(PolicyBase):
    MAPPING = "exp"
    """
    Exponentially decreasing learning rate:

    :math:`LR = LR_{base} \\gamma^{a\\,iter}`
    """
    def __init__(self, lr_to_adjust, **kwargs):
        super(ExpPolicy, self).__init__(**kwargs)
        self.base_lr = kwargs.get("base_lr", lr_to_adjust)
        self.gamma = kwargs["gamma"]
        self.a_ratio = kwargs["a_ratio"]

    def __call__(self, itr):
        return self.base_lr * (self.gamma ** (self.a_ratio * itr))


@implementer(ILRPolicy)
class FixedAjustPolicy(PolicyBase):
    MAPPING = "fixed"
    """
    Fixed learning rate:

    :math:`LR = LR_{base}`
    """
    def __init__(self, lr_to_adjust, **kwargs):
        super(FixedAjustPolicy, self).__init__(**kwargs)
        self.base_lr = kwargs.get("base_lr", lr_to_adjust)

    def __call__(self, itr):
        return self.base_lr


@implementer(ILRPolicy)
class StepExpPolicy(PolicyBase):
    MAPPING = "step_exp"
    """
    Step exponential decrease of learning_rate:

    :math:`LR = LR_{base} \\gamma^{floor(\\frac{iter}{step})}`
    """
    def __init__(self, lr_to_adjust, **kwargs):
        super(StepExpPolicy, self).__init__(**kwargs)
        self.base_lr = kwargs.get("base_lr", lr_to_adjust)
        self.gamma = kwargs["gamma"]
        self.step = kwargs["step"]

    def __call__(self, itr):
        return self.base_lr * (
            self.gamma ** floor(float(itr) / float(self.step)))


@implementer(ILRPolicy)
class InvAdjustPolicy(PolicyBase):
    MAPPING = "inv"
    """
    :math:`LR = LR_{base} \\dot (1 + \\gamma \\, iter) ^ {-pow}`
    """
    def __init__(self, lr_to_adjust, **kwargs):
        super(InvAdjustPolicy, self).__init__(**kwargs)
        self.base_lr = kwargs.get("base_lr", lr_to_adjust)
        self.gamma = kwargs["gamma"]
        self.pow_ratio = kwargs["pow_ratio"]

    def __call__(self, itr):
        return self.base_lr * (1.0 + self.gamma * itr) ** (-self.pow_ratio)


@implementer(ILRPolicy)
class ArbitraryStepPolicy(PolicyBase):
    MAPPING = "arbitrary_step"
    """
    Creates arbitrary step function.

    Arguments:
        base_lr: learning_rate to adjust (from kwargs or current learning_rate)
        lrs_with_lengths: list with tuples. First argument of tuple -\
        coefficition for leraning_rate, Second argument of tuple: number of\
        iterations with that learning_rate.\
        lrs_with_lengths = [(coeff1, N), (coeff2, M)]\
        lr = lr_to_adjust * coeff1 for N iterations\
        lr = lr_to_adjust * coeff2 for M iterations
    """
    def __init__(self, lr_to_adjust, **kwargs):
        super(ArbitraryStepPolicy, self).__init__(**kwargs)

        base_lr = kwargs.get("base_lr", lr_to_adjust)
        lrs_with_lengths = kwargs["lrs_with_lengths"]
        assert lrs_with_lengths is not None
        self.x_array = []
        self.y_array = []

        self.x_array.append(-1)
        self.y_array.append(base_lr * lrs_with_lengths[0][0])

        cur_iter = 0

        for coeff, length in lrs_with_lengths:
            assert coeff*base_lr >= 0
            assert length > 0
            self.x_array.append(cur_iter)
            self.y_array.append(coeff*base_lr)
            if length > 1:
                self.x_array.append(cur_iter + length - 1)
                self.y_array.append(coeff*base_lr)
            cur_iter += length

        self.out_function = interp1d(
            self.x_array, self.y_array, bounds_error=False, fill_value=0)

    def __call__(self, itr):
        return self.out_function(itr)

    def __getstate__(self):
        return self.x_array, self.y_array

    def __setstate__(self, state):
        self.x_array, self.y_array = state
        self.out_function = interp1d(
            self.x_array, self.y_array, bounds_error=False, fill_value=0)
