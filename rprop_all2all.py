"""
Created on Sep 16, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.

RPROP for :class:`veles.znicz.All2All`
"""

import numpy

from veles import memory
from veles.znicz.gd import GradientDescent


class RPropAll2All(GradientDescent):
    MAPPING = {"rprop_all2all"}
    """
    Only CPU version is implemented
    """
    def __init__(self, workflow, **kwargs):
        super(RPropAll2All, self).__init__(workflow, **kwargs)
        self.initial_learning_rate = 0.01
        self.min_learning_rate = 10 ** -6
        self.max_learning_rate = 1
        self.increase = 1.05
        self.decrease = 0.80

        self.weight_lrs = memory.Vector()
        self.bias_lrs = memory.Vector()

    def initialize(self, device, **kwargs):
        super(RPropAll2All, self).initialize(device=device, **kwargs)
        self.weight_lrs.mem = numpy.zeros(
            shape=self.weights.mem.shape, dtype=self.weights.mem.dtype)
        self.bias_lrs.mem = numpy.zeros(
            shape=self.bias.mem.shape, dtype=self.bias.mem.dtype)

        self.weight_lrs.initialize(self.device)
        self.bias_lrs.initialize(self.device)

    def cpu_weights_update(self):
        self.input.map_read()
        self.err_output.map_read()
        self.weights.map_write()
        self.gradient_weights.map_write()
        gradient = numpy.dot(
            self.err_output.mem.swapaxes(0, 1),
            self.input.mem.reshape((self.input.mem.shape[0],
                                    numpy.prod(self.input.mem.shape[1:]))))

        grad_sign = numpy.sign(gradient)
        grad_delta_sign = numpy.sign(self.gradient_weights.mem * gradient)

        increase_ratios = numpy.where(grad_delta_sign > 0, self.increase, 1)
        decrease_ratios = numpy.where(grad_delta_sign < 0, self.decrease, 1)

        self.weight_lrs.mem *= increase_ratios
        self.weight_lrs.mem * decrease_ratios

        self.weight_lrs.mem[:] = self.weight_lrs.mem.clip(
            self.min_learning_rate, self.max_learning_rate)[:]

        if self.weights_transposed:
            self.weights.mem -= (grad_sign * self.weight_lrs.mem).transpose()
        else:
            self.weights.mem -= grad_sign * self.weight_lrs.mem

        self.gradient_weights.mem[:] = gradient[:]

    def cpu_bias_update(self):
        if not self.include_bias:
            return

        self.err_output.map_read()
        self.bias.map_write()
        self.gradient_bias.map_write()

        gradient = numpy.sum(self.err_output.mem, axis=0)
        grad_sign = numpy.sign(gradient)
        grad_delta_sign = numpy.sign(self.gradient_bias.mem * gradient)
        increase_ratios = numpy.where(grad_delta_sign > 0, self.increase, 1)
        decrease_ratios = numpy.where(grad_delta_sign < 0, self.decrease, 1)
        self.bias_lrs.mem *= increase_ratios
        self.bias_lrs.mem * decrease_ratios

        self.bias_lrs.mem[:] = self.bias_lrs.mem.clip(
            self.min_learning_rate, self.max_learning_rate)[:]

        self.bias.mem -= grad_sign * self.bias_lrs.mem
        self.gradient_bias.mem[:] = gradient[:]

    def ocl_weights_update(self):
        pass

    def ocl_bias_update(self):
        pass

    def ocl_run(self):
        # TODO(a.golovizin): implement OCL version
        self.cpu_run()
