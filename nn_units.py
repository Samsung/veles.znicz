"""
Created on Jan 28, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""
import numpy
from units import OpenCLUnit
import rnd
import config
import formats


class Forward(OpenCLUnit):
    """Base class for forward propagation units.

    Attributes:
        input: input layer values.
        output: output layer values.
        weights: weights.
        bias: bias.
        weights_magnitude: magnitude of the random distribution of weights.
        rand: rnd.Rand() object for initial weights generation.
    """
    def __init__(self, workflow, **kwargs):
        weights_magnitude = kwargs.get("weights_magnitude")
        rand = kwargs.get("rand", rnd.default)
        weights_transposed = kwargs.get("weights_transposed", False)
        kwargs["weights_magnitude"] = weights_magnitude
        kwargs["rand"] = rand
        kwargs["weights_transposed"] = weights_transposed
        kwargs["view_group"] = kwargs.get("view_group", "WORKER")
        super(Forward, self).__init__(workflow, **kwargs)
        self.input = None
        self.output = formats.Vector()
        self.weights = formats.Vector()
        self.bias = formats.Vector()
        self.weights_magnitude = weights_magnitude
        self.rand = rand
        self.weights_transposed = weights_transposed
        self.exports = ["weights", "bias", "weights_transposed"]

    def generate_data_for_slave(self, slave=None):
        self.workflow.lock_data()
        self.weights.map_read()
        self.bias.map_read()
        data = (self.weights.v.copy(), self.bias.v.copy())
        self.workflow.unlock_data()
        return data

    def apply_data_from_master(self, data):
        self.weights.map_invalidate()
        self.bias.map_invalidate()
        numpy.copyto(self.weights.v, data[0])
        numpy.copyto(self.bias.v, data[1])


class GD(OpenCLUnit):
    """Base class for gradient descent units.

    Attributes:
        h: input layer values.
        y: output layer values.
        err_y: error to backpropagate.
        err_h: backpropagated error.
        weights: weights.
        bias: bias.
        batch_size: current minibatch size.
        global_alpha: gradient descent speed (positive).
        global_lambda: coefficient (positive or zero) for weights
                       regularization term (lambda/2 * sum(weights^2)).
        batch_size: effective batch size (if None, get it from y).
        weights_transposed: assume weights matrix as a transposed one.
        store_gradient: will save gradient as separate Vector().
        apply_gradient: will apply gradient.
    """
    def __init__(self, workflow, **kwargs):
        global_alpha = kwargs.get("global_alpha", 0.01)
        global_lambda = kwargs.get("global_lambda", 0.00005)
        weights_transposed = kwargs.get("weights_transposed", False)
        store_gradient = kwargs.get("store_gradient", workflow.is_slave)
        apply_gradient = kwargs.get("apply_gradient", not workflow.is_slave)
        kwargs["global_alpha"] = global_alpha
        kwargs["global_lambda"] = global_lambda
        kwargs["weights_transposed"] = weights_transposed
        kwargs["store_gradient"] = store_gradient
        kwargs["apply_gradient"] = apply_gradient
        kwargs["view_group"] = kwargs.get("view_group", "TRAINER")
        super(GD, self).__init__(workflow, **kwargs)
        self.h = None
        self.y = None
        self.err_y = None  # formats.Vector()
        self.err_h = formats.Vector()
        self.weights = None
        self.bias = None
        self.batch_size = None  # [0]
        self.global_alpha = global_alpha
        self.global_lambda = global_lambda
        self.weights_transposed = weights_transposed
        self.store_gradient = store_gradient
        self.apply_gradient = apply_gradient
        self.gradient_weights = formats.Vector()
        self.gradient_bias = formats.Vector()

    def generate_data_for_slave(self, slave=None):
        return (self.global_alpha, self.global_lambda)

    def apply_data_from_master(self, data):
        self.global_alpha = data[0]
        self.global_lambda = data[1]
        if self.gradient_weights.v is None or self.gradient_bias.v is None:
            return
        self.gradient_weights.map_invalidate()
        self.gradient_weights.v[:] = 0
        self.gradient_bias.map_invalidate()
        self.gradient_bias.v[:] = 0

    def generate_data_for_master(self):
        if (not self.run_executed or
            self.gradient_weights.v is None or self.gradient_bias.v is None):
            return None
        self.run_executed = False
        self.gradient_weights.map_read()
        self.gradient_bias.map_read()
        return (self.gradient_weights.v, self.gradient_bias.v)

    def apply_data_from_slave(self, data, slave=None):
        self.workflow.lock_data()
        self.weights.map_write()
        self.bias.map_write()
        self.weights.v += data[0]
        self.bias.v += data[1]
        self.workflow.unlock_data()
