"""
Created on Mar 12, 2013

Filters in data stream neural network model.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
from numpy import *


class GeneralFilter(object):
    """General filter in data stream neural network model.
    """
    def __init__(self):
        super().__init__()


class PrimitiveFilter(GeneralFilter):
    """Filter that cannot contain other filters.
    """
    def __init__(self):
        super().__init__()


class ErrExists(Exception):
    """Exception, raised when something already exists in the set.
    """
    pass


class ErrNotExists(Exception):
    """Exception, raised when something does not exist in the set.
    """
    pass


class ContainerFilter(GeneralFilter):
    """Filter that contains other filters.
    
    Attributes:
        filters: set of filters in the container.
        links: dictionary of sets of filters
    """
    def __init__(self):
        super().__init__()
        self.filters = set()
        self.links = {}
    
    def add(self, f):
        """Adds filter to the container.
        
        Args:
            f: filter to add.
        
        Returns:
            f.
        
        Raises:
            ErrExists.
        """
        if f in self.filters:
            raise ErrExists
        self.filters.add(f)
        return f

    def link(self, src, dst):
        """Links to filters
        
        Args:
            src: source filter
            dst: destination filter
        
        Returns:
            dst.
        
        Raises:
            ErrNotExists
            ErrExists
        """
        if (src not in self.filters) or (dst not in self.filters):
            raise ErrNotExists
        if src not in self.links:
            self.links[src] = set()
        if dst in self.links[src]:
            raise ErrExists;
        self.links[src].add(dst)


class All2AllFilter(PrimitiveFilter):
    """Classical MLP layer to layer connection.
    
    Attributes:
        weights: matrix of weights.
    """
    def __init__(self):
        super().__init__()
        self.weights = empty((1), dtype=float32)
