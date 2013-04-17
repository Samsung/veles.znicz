"""
Created on Apr 15, 2013

Data formats for connectors.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import pyopencl
import opencl
import os


CPU = 1
GPU = 2


class OpenCLConnector(filters.Connector):
    """Connector that uses OpenCL.

    Attributes:
        device: OpenCL device.
        what_changed: what buffer has changed?
    """
    def __init__(self, device = None, unpickling = 0):
        super(OpenCLConnector, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.device = device
        self.what_changed = 0

    def gpu_2_cpu(self):
        """Copies buffer from GPU to CPU.
        """
        self.what_changed = 0

    def cpu_2_gpu(self):
        """Copies buffer from CPU to GPU.
        """
        self.what_changed = 0

    def __getstate__(self):
        """Get data from OpenCL device before pickling.
        """
        if self.device and (self.what_changed & GPU) and self.device.pid == os.getpid():
            self.gpu_2_cpu()
        return super(OpenCLConnector, self).__getstate__()

    def update(self, what_changed = CPU):
        """Updates data ready status.

        Parameters:
            cpu: cpu buffer changed.
            gpu: gpu buffer changed.
        """
        self.what_changed = what_changed
        super(OpenCLConnector, self).update()

    def sync(self, what_will_use = CPU):
        if not self.device:
            return
        if (what_will_use & CPU) and (self.what_changed & GPU):
            self.gpu_2_cpu()
            return
        if (what_will_use & GPU) and (self.what_changed & CPU):
            self.cpu_2_gpu()
            return

    def initialize(self, device = None):
        """Create OpenCL buffer handle here.
        """
        pass

    def _map_wait_unmap(self, buf, buf_, OP):
        arr, event = pyopencl.enqueue_map_buffer(queue=self.device.queue_, buf=buf_, flags=OP, \
            offset=0, shape=buf.shape, dtype=buf.dtype, order="C", wait_for=None, is_blocking=False)
        event.wait()
        arr.base.release(queue=self.device.queue_)
        self.what_changed = 0

    def _buffer(self, buf):
        mf = pyopencl.mem_flags
        return pyopencl.Buffer(self.device.context_, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=buf)


class Batch(OpenCLConnector):
    """Batch.

    Attributes:
        batch: numpy array with first dimension as batch.
        batch_: OpenCL buffer mapped to batch.
    """
    def __init__(self, device = None, unpickling = 0):
        super(Batch, self).__init__(device=device, unpickling=unpickling)
        self.batch_ = None
        if unpickling:
            self.batch = filters.realign(self.batch)
            return
        self.batch = None

    def initialize(self, device = None):
        if self.batch_:
            return
        if device:
            self.device = device
        if not self.device:
            return
        self.batch_ = self._buffer(self.batch)

    def gpu_2_cpu(self):
        self._map_wait_unmap(self.batch, self.batch_, opencl.CL_MAP_READ)

    def cpu_2_gpu(self):
        self._map_wait_unmap(self.batch, self.batch_, opencl.CL_MAP_WRITE_INVALIDATE_REGION)


class Vector(OpenCLConnector):
    """Vector.

    Attributes:
        v: numpy array as a vector.
        v_: OpenCL buffer mapped to vector.
    """
    def __init__(self, device = None, unpickling = 0):
        super(Vector, self).__init__(device=device, unpickling=unpickling)
        self.v_ = None
        if unpickling:
            self.v = filters.realign(self.v)
            return
        self.v = None

    def initialize(self, device = None):
        if self.v_:
            return
        if device:
            self.device = device
        if not self.device:
            return
        self.v_ = self._buffer(self.v)

    def gpu_2_cpu(self):
        self._map_wait_unmap(self.v, self.v_, opencl.CL_MAP_READ)

    def cpu_2_gpu(self):
        self._map_wait_unmap(self.v, self.v_, opencl.CL_MAP_WRITE_INVALIDATE_REGION)


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
