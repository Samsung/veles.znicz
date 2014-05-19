# encoding: utf-8

"""
Created on May 16, 2014

**Dynamic adjust of learning rates of GD units.**

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import veles.units as units
from veles.znicz.nn_units import GradientDescentBase

from math import floor

class LearningRateAdjust(units.Unit):
    """
    This unit should be linked from Decision to run with each minibatch.
    Does nothing if ``lr_function`` is not set.

    Args:
        lr_function(:class:`function`): a function that takes :class:`int`
            iteration number and returns :class:`float` learning rate
    """
    def __init__(self, workflow, **kwargs):
        self._lr_function = kwargs.get("lr_function", None)
        self._gradient_units = []
        self._minibatches_count = 0
        super(LearningRateAdjust, self).__init__(workflow, **kwargs)

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

    def run(self):
        """
        Adjusts learning rates of GD units according to ``lr_function``
        Should be run every minibatch before GD units.
        """
        if self._lr_function is not None:
            learning_rate = self._lr_function(self._minibatches_count)
            for gd_elm in self._gradient_units:
                gd_elm.learning_rate = learning_rate

        self._minibatches_count += 1

#LEARNING RATE POLICIES:

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
