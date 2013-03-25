"""
Created on Mar 18, 2013

Classes for custom exceptions

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
class ErrNotExists(Exception):
    """Exception, raised when something does not exist.
    """
    pass


class ErrExists(Exception):
    """Exception, raised when something already exists.
    """
    pass


class ErrNotImplemented(Exception):
    """Exception, raised when something is not implemented.
    """
    pass


class ErrBadFormat(Exception):
    """Exception, raised when bad format or data occured somethere.
    """
    pass


class ErrOpenCL(Exception):
    """Exception, raised when OpenCL event returns a error
    
    Attributes:
        event: pyopencl Event() object.
    """
    def __init__(self, event):
        super(ErrOpenCL, self).__init__()
        self.event = event
