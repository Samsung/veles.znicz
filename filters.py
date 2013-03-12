"""
Created on Mar 12, 2013

Filters in data stream neural network model

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
from numpy  import *


class GeneralFilter(object):
    """GeneralFilter
    
    General filter in data stream neural network model
    
    Attributes:
    """
    def __init__(self):
        pass


class All2AllFilter(GeneralFilter):
    """All2AllFilter
    
    Classical MLP layer to layer connection
    
    Attributes:
        weights: matrix of weights
    """
    def __init__(self):
        super().__init__()
        self.weights = empty((1), dtype=float32)
