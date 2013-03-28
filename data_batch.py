"""
Created on Mar 21, 2013

Output formats and some filters for data batches.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters


class DataBatch(filters.OutputData):
    """Base data batch class.
    
    Attributes:
        data: numpy array where distinct elements of the batch reside in the first dimension.
        labels: labels for the elements in the batch.
        device: opencl.Device() object.
        data_: opencl buffer.
    """
    def __init__(self, unpickling = 0, data = None, labels = None):
        super(DataBatch, self).__init__(unpickling)
        self.data_ = None
        if unpickling:
            return
        self.data = data
        self.labels = labels
        self.device = None
