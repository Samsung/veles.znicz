"""
Created on Mar 20, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters


class All2All(filters.GeneralFilter):
    """All2All layer to layer
    """
    def __init__(self, parent):
        super(All2All, self).__init__(parent)

    def from_batch2D(self):
        print("from_batch2D")
        #TODO(a.kazantsev): notify parent on completion (OpenCL event)
        #if self.parent:
        #    self.parent.output_changed(self)

    def input_changed(self, src):
        if src.output.__class__.__name__ == "DataBatch2D":
            return self.from_batch2D()
        return 
