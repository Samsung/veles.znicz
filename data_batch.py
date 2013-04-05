"""
Created on Mar 21, 2013

Output formats and some filters for data batches.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters


class Labels(filters.State):
    """Labels for data batch.
    
    Attributes:
        data: array of labels numbered from 0 to n_classes - 1.
        n_classes: number of classes for data batch.
    """
    def __init__(self, unpickling = 0, data = None, n_classes = 0):
        super(Labels, self).__init__(unpickling)
        if unpickling:
            return
        self.data = data
        self.n_classes = n_classes


class DataBatch(filters.State):
    """Base data batch class.
    
    Attributes:
        data: numpy array where distinct elements of the batch reside in the first dimension.
        labels: labels for the elements in the batch.
        device: opencl.Device() object.
        data_: opencl buffer.
    """
    def __init__(self, unpickling = 0, data = None):
        super(DataBatch, self).__init__(unpickling)
        self.data_ = None
        if unpickling:
            return
        self.data = data
        self.device = None
