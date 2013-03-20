"""
Created on Mar 19, 2013

Classes for batch input/output data.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import numpy


class DataBatch(filters.OutputData):
    """Batch data.
    
    Attributes:
        n_samples: number of samples in batch
    """
    def __init__(self, n_samples):
        super(DataBatch, self).__init__()
        self.n_samples = n_samples


class DataBatch2D(DataBatch):
    """Batch data with 2D gray images as floats normalized to [0, 1] range.
    
    Attributes:
        width: width of an image.
        height: height of an image.
        labels: labels for each image.
        data: array of floats.
    """
    def __init__(self, n_samples, width, height, labels):
        super(DataBatch2D, self).__init__(n_samples)
        self.width = width
        self.height = height
        self.labels = labels
        self.data = numpy.zeros((self.n_samples, self.height, self.width), dtype=numpy.float32)


class DataBatchConvolutions(DataBatch):
    """Batch data with convolutions results for one layer.
    
    Attributes:
        width: width of the filter.
        height: height of the filter.
        n_filters: number of filters.
        data: array of floats.
    """
    def __init__(self, n_samples, width, height, n_filters):
        super(DataBatchConvolutions, self).__init__(n_samples)
        self.width = width
        self.height = height
        self.n_filters = n_filters
        #TODO(a.kazantsev): the following line is a stub, change it
        self.data = numpy.zeros((self.n_samples, self.height, self.width), dtype=numpy.float32)
