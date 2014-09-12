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
from zope.interface import implementer

from veles.units import IUnit, Unit
from veles.znicz.nn_units import GradientDescentBase
from veles.distributable import IDistributable


@implementer(IUnit, IDistributable)
class LearningRateAdjust(Unit):
    """
    This unit should be linked from Decision to run with each minibatch.
    Does nothing if ``lr_function`` is not set.

    Args:
        lr_function(:class:`function`): a function that takes `int`
            iteration number and returns :class:`float` **weight** learning
            rate
        bias_lr_function(:class:`function`): a function that takes `int`
            iteration number and returns :class:`float` **bias** learning rate
            (if nothing is set - `lr_function` is taken)
    """

    def __init__(self, workflow, **kwargs):
        super(LearningRateAdjust, self).__init__(workflow, **kwargs)
        self._gd_units = []
        self._minibatches_count = 0

    def add_gd_unit(self, gd_unit, lr_function, bias_lr_function):
        """
        Gradient unit should have learning_rate property.

        Args:
            grad_unit(:class:`GradientDescentBase`): gradient unit with
                ``learning_rate`` parameter to manipulate.
        """
        assert isinstance(gd_unit, GradientDescentBase)
        self._gd_units.append((gd_unit, lr_function, bias_lr_function))

    def initialize(self, **kwargs):
        pass

    def run(self):
        """
        Adjusts learning rates of GD units according to ``lr_function``
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
                        self.info("LR: %.4e => %.4e",
                                  gd_unit.learning_rate, lr)
                    gd_unit.learning_rate = lr
            if bias_lr_func is not None:
                lr = float(bias_lr_func(self._minibatches_count))
                if gd_unit.learning_rate_bias != lr:
                    if not notified:
                        notified = True
                        self.info("LRB: %.4e => %.4e",
                                  gd_unit.learning_rate_bias, lr)
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


def exp_adjust_policy(base_lr, gamma, a_ratio):
    """
    Creates exponentially decreasing learning ratio policy:

    :math:`LR = LR_{base} \\gamma^{a\\,iter}`

    Returns:
        :class:`function(itr)`
    """
    return lambda itr: base_lr * (gamma ** (a_ratio * itr))


def fixed_adjust_policy(base_lr):
    """
    Creates fixed learning rate policy

    :math:`LR = LR_{base}`

    Returns:
        :class:`function(iter)`
    """
    return lambda itr: base_lr


def step_exp_adjust_policy(base_lr, gamma, step):
    """
    Creates step exponential decrease of LR policy
    :math:`LR = LR_{base} \\gamma^{floor(\\frac{iter}{step})}`

    Returns:
        :class:`function(itr)`
    """
    return lambda itr: base_lr * gamma ** floor(float(itr) / float(step))


def inv_adjust_policy(base_lr, gamma, pow_ratio):
    """
    :math:`LR = LR_{base} \\dot (1 + \\gamma \\, iter) ^ {-pow}`

    Returns:
        :class:`function(itr)`
    """
    return lambda itr: base_lr * (1.0 + gamma * itr) ** (-pow_ratio)


def arbitrary_step_policy(lrs_with_lengths):
    """
    Creates arbitrary step function: LR1 for N iters, LR2 for next M iters, etc

    For example: arbitrary_step_function_policy([(0.5, 5), (0.3, 3), (0.1, 1)]

    Args:
        lrs_with_weights(list): a list of `(length, lr)` tuples that describes
            which learning rate should be set for each number of iterations,
            one by one.
    Returns:
        :class:`function(itr)`: this function returns 0 when last length ends
    """
    assert lrs_with_lengths is not None

    x_array = []
    y_array = []

    x_array.append(-1)
    y_array.append(lrs_with_lengths[0][0])

    cur_iter = 0

    for lr, length in lrs_with_lengths:
        assert lr >= 0
        assert length > 0
        x_array.append(cur_iter)
        y_array.append(lr)
        if length > 1:
            x_array.append(cur_iter + length - 1)
            y_array.append(lr)
        cur_iter += length

    out_function = interp1d(x_array, y_array, bounds_error=False, fill_value=0)
    return out_function
