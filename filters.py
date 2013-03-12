"""
Created on Mar 12, 2013

Filters in data stream neural network model

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
from numpy import *


class GeneralFilter(object):
    """GeneralFilter
    
    General filter in data stream neural network model
    
    Attributes:
    """
    def __init__(self):
        super().__init__()


class PrimitiveFilter(GeneralFilter):
    """PrimitiveFilter
    
    Filter that cannot contain other filters
    """
    def __init__(self):
        super().__init__()


class FilterExists(Exception):
    """Exception, raised when filter exists in the container
    """
    pass


class ContainerFilter(GeneralFilter):
    """ContainerFilter
    
    Filter that contains other filters
    
    Attributes:
        filters: set of filters in the container
    """
    def __init__(self):
        super().__init__()
        self.filters = set()
    
    def check_exists(self, f):
        if f in self.filters:
            raise FilterExists
    
    def add(self, f):
        """add
        
        Adds filter to the container
        
        Args:
            f: filter to add
        
        Returns:
            f
        
        Raises:
            FilterExists
        """
        self.check_exists(f)
        self.filters.add(f)
        return f


class All2AllFilter(PrimitiveFilter):
    """All2AllFilter
    
    Classical MLP layer to layer connection
    
    Attributes:
        weights: matrix of weights
    """
    def __init__(self):
        super().__init__()
        self.weights = empty((1), dtype=float32)
