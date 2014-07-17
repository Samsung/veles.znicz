# encoding: utf-8

"""
Created on May 16, 2014

**Dynamic adjust of learning rates of GD units.**

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
        self._lr_function = kwargs.get("lr_function", None)
        self._bias_lr_function = kwargs.get("bias_lr_function",
                                            self._lr_function)
        self._gradient_units = []
        self._minibatches_count = 0
        self._prev_lr = 1.0e30
        self._prev_bias_lr = 1.0e30

    def add_one_gd_unit(self, grad_unit):
        """
        Gradient unit should have learning_rate property.

        Args:
            grad_unit(:class:`GradientDescentBase`): gradient unit with
                ``learning_rate`` parameter to manipulate.
        """
        assert isinstance(grad_unit, GradientDescentBase)
        self._gradient_units.append(grad_unit)

    def add_gd_units(self, grad_units):
        """
        Args:
            grad_units(iterable): gradient units to add. Skips all except
                instances of :class:`GradientDescentBase`
        """
        for gd_unit in grad_units:
            if isinstance(gd_unit, GradientDescentBase):
                self.add_one_gd_unit(gd_unit)

    def initialize(self, **kwargs):
        pass

    def run(self):
        """
        Adjusts learning rates of GD units according to ``lr_function``
        Should be run every minibatch before GD units.
        """
        if self._lr_function is not None:
            learning_rate = self._lr_function(self._minibatches_count)
            if learning_rate != self._prev_lr:
                self._prev_lr = learning_rate
                for gd_elm in self._gradient_units:
                    gd_elm.learning_rate = learning_rate

        if self._bias_lr_function is not None:
            bias_lr = self._bias_lr_function(self._minibatches_count)
            if bias_lr != self._prev_bias_lr:
                self._prev_bias_lr = learning_rate
                for gd_elm in self._gradient_units:
                    gd_elm.learning_rate_bias = bias_lr

        self._minibatches_count += 1

    # IDistributable implementation
    def generate_data_for_slave(self, slave):
        data = (self._minibatches_count, self._lr_function,
                self._bias_lr_function)
        return data

    def generate_data_for_master(self):
        return None

    def apply_data_from_master(self, data):
        self._minibatches_count, self._lr_function, self._bias_lr_function = \
            data

    def apply_data_from_slave(self, data, slave):
        pass

    def drop_slave(self, slave):
        pass


# LEARNING RATE POLICIES:


def exp_adjust_policy(base_lr, gamma, a_ratio):
    """
    Creates exponentially decreasing learning ratio policy:

    :math:`LR = LR_{base} \\gamma^{a\\,iter}`

    Returns:
        :class:`function(iter)`
    """
    return lambda iter: base_lr * (gamma ** (a_ratio * iter))


def fixed_adjust_policy(base_lr):
    """
    Creates fixed learning rate policy

    :math:`LR = LR_{base}`

    Returns:
        :class:`function(iter)`
    """
    return lambda iter: base_lr


def step_exp_adjust_policy(base_lr, gamma, step):
    """
    Creates step exponential decrease of LR policy
    :math:`LR = LR_{base} \\gamma^{floor(\\frac{iter}{step})}`

    Returns:
        :class:`function(iter)`
    """
    return lambda iter: base_lr * gamma ** floor(float(iter) / float(step))


def inv_adjust_policy(base_lr, gamma, pow_ratio):
    """
    :math:`LR = LR_{base} \\dot (1 + \\gamma \\, iter) ^ {-pow}`

    Returns:
        :class:`function(iter)`
    """
    return lambda iter: base_lr * (1.0 + gamma * iter) ** (-pow_ratio)


def arbitrary_step_policy(lrs_with_lengths):
    """
    Creates arbitrary step function: LR1 for N iters, LR2 for next M iters, etc

    For example: arbitrary_step_function_policy([(0.5, 5), (0.3, 3), (0.1, 1)]

    Args:
        lrs_with_weights(list): a list of `(length, lr)` tuples that describes
            which learning rate should be set for each number of iterations,
            one by one.
    Returns:
        :class:`function(iter)`: this function returns 0 when last length ends
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
