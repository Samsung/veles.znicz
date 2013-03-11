"""
Created on Mar 11, 2013

One layer of the neural network.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
from numpy  import *


class Layer(object):
    """One Layer of the neural network.
    
    One layer of the neural network.
    
    Attributes:
        weights: array of weights
    """
    def __init__(self):
        self.weights = empty((2, 25), dtype=float32)
