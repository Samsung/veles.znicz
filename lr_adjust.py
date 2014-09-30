# encoding: utf-8
"""
Created on May 16, 2014

Dynamic adjust of learning rates of GD units. Learning rate are changed
    according to iteration number, each iteration.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
from __future__ import division

from math import floor
from scipy.interpolate import interp1d
from zope.interface import implementer, Interface

from veles.units import IUnit, Unit
from veles.znicz.nn_units import GradientDescentBase
from veles.distributable import IDistributable


@implementer(IUnit, IDistributable)
class LearningRateAdjust(Unit):
    """
    This unit should be linked from Decision to run with each minibatch.
    """

    def __init__(self, workflow, **kwargs):
        super(LearningRateAdjust, self).__init__(workflow, **kwargs)
        self._gd_units = []
        self._minibatches_count = 0

    def add_gd_unit(self, gd_unit, lr_policy, bias_lr_policy):
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
        self._gd_units.append((gd_unit, lr_policy, bias_lr_policy))

    def initialize(self, **kwargs):
        pass

    def run(self):
        """
        Adjusts learning rates of GD units according to ``lr_policy``
        Should be run every minibatch before GD units.
        """
        if self.is_slave:
            return

        notified = False

        for gd_unit, lr_func, bias_lr_func in self._gd_units:
            if lr_func is not None:
                lr = float(lr_func(self._minibatches_count))
                if gd_unit.learning_rate != lr:
                    if not notified:
                        notified = True
                    gd_unit.learning_rate = lr
            if bias_lr_func is not None:
                lr = float(bias_lr_func(self._minibatches_count))
                if gd_unit.learning_rate_bias != lr:
                    if not notified:
                        notified = True
                    gd_unit.learning_rate_bias = lr

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


@implementer(ILRPolicy)
class ExpPolicy(object):
    """
    Exponentially decreasing learning rate:

    :math:`LR = LR_{base} \\gamma^{a\\,iter}`
    """
    def __init__(self, base_lr, gamma, a_ratio):
        self.base_lr = base_lr
        self.gamma = gamma
        self.a_ratio = a_ratio

    def __call__(self, itr):
        return self.base_lr * (self.gamma ** (self.a_ratio * itr))


@implementer(ILRPolicy)
class FixedAjustPolicy(object):
    """
    Fixed learning rate:

    :math:`LR = LR_{base}`
    """
    def __init__(self, base_lr):
        self.base_lr = base_lr

    def __call__(self, itr):
        return self.base_lr


@implementer(ILRPolicy)
class StepExpPolicy(object):
    """
    Step exponential decrease of LR:

    :math:`LR = LR_{base} \\gamma^{floor(\\frac{iter}{step})}`
    """
    def __init__(self, base_lr, gamma, step):
        self.base_lr = base_lr
        self.gamma = gamma
        self.step = step

    def __call__(self, itr):
        return self.base_lr * (
            self.gamma ** floor(float(itr) / float(self.step)))


@implementer(ILRPolicy)
class InvAdjustPolicy(object):
    """
    :math:`LR = LR_{base} \\dot (1 + \\gamma \\, iter) ^ {-pow}`
    """
    def __init__(self, base_lr, gamma, pow_ratio):
        self.base_lr = base_lr
        self.gamma = gamma
        self.pow_ratio = pow_ratio

    def __call__(self, itr):
        return self.base_lr * (1.0 + self.gamma * itr) ** (-self.pow_ratio)


@implementer(ILRPolicy)
class ArbitraryStepPolicy(object):
    """
    Creates arbitrary step function: LR1 for N iters, LR2 for next M iters, etc

    For example: ArbitraryStepPolicy([(0.5, 5), (0.3, 3), (0.1, 1)])

    """
    def __init__(self, lrs_with_lengths):
        """
        Args:
        lrs_with_weights(list): a list of `(length, lr)` tuples that describes
            which learning rate should be set for each number of iterations,
            one by one.
        """
        assert lrs_with_lengths is not None
        self.x_array = []
        self.y_array = []

        self.x_array.append(-1)
        self.y_array.append(lrs_with_lengths[0][0])

        cur_iter = 0

        for lr, length in lrs_with_lengths:
            assert lr >= 0
            assert length > 0
            self.x_array.append(cur_iter)
            self.y_array.append(lr)
            if length > 1:
                self.x_array.append(cur_iter + length - 1)
                self.y_array.append(lr)
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
