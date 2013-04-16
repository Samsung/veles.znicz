"""
Created on Apr 15, 2013

Data formats for connectors.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import pyopencl
import opencl
import os


class Batch(filters.Connector):
    """Batch.

    Attributes:
        batch: numpy array with first dimension as batch.
        batch_: OpenCL buffer mapped to batch.
        device: OpenCL device.
    """
    def __init__(self, device = None, unpickling = 0):
        super(Batch, self).__init__(unpickling=unpickling)
        self.batch_ = None
        if unpickling:
            self.batch = filters.realign(self.batch)
            return
        self.device = device
        self.batch = None

    def read_buffer(self):
        arr, event = pyopencl.enqueue_map_buffer(queue=self.device.queue_, buf=self.batch_, \
                flags=opencl.CL_MAP_READ, offset=0, shape=self.batch.shape, \
                dtype=self.batch.dtype, order="C", wait_for=None, is_blocking=False)
        event.wait()
        arr.base.release(queue=self.device.queue_)

    def __getstate__(self):
        """Get data from OpenCL device before pickling.
        """
        if self.device and self.batch_ and self.device.pid == os.getpid():
            self.read_buffer()
        return super(Batch, self).__getstate__()


class Vector(filters.Connector):
    """Vector.
    
    Attributes:
        v: numpy array.
        v_: OpenCL buffer mapped to v.
        device: OpenCL device.
    """
    def __init__(self, device = None, unpickling = 0):
        super(Vector, self).__init__(unpickling=unpickling)
        self.v_ = None
        if unpickling:
            self.v = filters.realign(self.v)
            return
        self.device = device
        self.v = None

    def read_buffer(self):
        arr, event = pyopencl.enqueue_map_buffer(queue=self.device.queue_, buf=self.v_, \
                flags=opencl.CL_MAP_READ, offset=0, shape=self.v.shape, \
                dtype=self.v.dtype, order="C", wait_for=None, is_blocking=False)
        event.wait()
        arr.base.release(queue=self.device.queue_)

    def __getstate__(self):
        """Get data from OpenCL device before pickling.
        """
        if self.device and self.v_ and self.device.pid == os.getpid():
            self.read_buffer()
        return super(Vector, self).__getstate__()


class Labels(filters.Connector):
    """Labels for batch.

    Attributes:
        n_classes: number of classes.
        v: numpy array sized as a batch with one label per image in range [0, n_classes).
    """
    def __init__(self, unpickling = 0):
        super(Labels, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.n_classes = 0
        self.v = None
