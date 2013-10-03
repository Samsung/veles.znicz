"""
Created on Aug 27, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units


class Conv(units.OpenCLUnit):
    """Convolutional layer.
    """
    def __init__(self, device=None):
        super(Conv, self).__init__(device=device)

    def initialize(self):
        pass

    def cpu_run(self):
        return self.gpu_run()

    def gpu_run(self):
        pass
