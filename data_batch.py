"""
Created on Mar 21, 2013

Output formats for data batches.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters


class DataBatch(filters.OutputData):
    """Base data batch class.
    
    Attributes:
        data: numpy array where distinct elements of the batch reside in the first dimension.
        labels: labels for the elements in the batch. 
    """
    def __init__(self, data, labels):
        super(DataBatch, self).__init__()
        self.data = data
        self.labels = labels
