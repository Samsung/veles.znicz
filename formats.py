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

    def initialize(self, device = None):
        if self.batch_:
            return
        if device:
            self.device = device
        if not self.device:
            return
        mf = pyopencl.mem_flags
        self.batch_ = pyopencl.Buffer(self.device.context_, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.batch)

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

    def initialize(self, device = None):
        if self.v_:
            return
        if device:
            self.device = device
        if not self.device:
            return
        mf = pyopencl.mem_flags
        self.v_ = pyopencl.Buffer(self.device.context_, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.v)

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


class Labels(Batch):
    """Labels for batch.

    Attributes:
        n_classes: number of classes.
    """
    def __init__(self, device = None, unpickling = 0):
        super(Labels, self).__init__(device=device, unpickling=unpickling)
        if unpickling:
            return
        self.n_classes = 0
