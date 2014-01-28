"""
Created on Jan 28, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""

from units import OpenCLUnit


class Forward(OpenCLUnit):
    """Base class for forward propagation units.

    Attributes:
        input: input layer values.
        output: output layer values.
        weights: weights.
        bias: bias.
    """
    def __init__(self, device=None):
        super(Forward, self).__init__(device=device)
        self.input = None
        self.output = None
        self.weights = None
        self.bias = None
        self.exports = ["weights", "bias"]


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
    """
    def __init__(self, device=None):
        super(GD, self).__init__(device=device)
        self.h = None
        self.y = None
        self.err_y = None
        self.err_h = None
        self.weights = None
        self.bias = None
        self.batch_size = None