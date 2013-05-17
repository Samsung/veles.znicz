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
        arr_: first argument returned by pyopencl.enqueue_map_buffer().
        aligned_: numpy array aligned to device.info.BLOCK_SIZE.
    """
    def __init__(self, device = None, unpickling = 0):
        super(OpenCLConnector, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.device = device
        self.what_changed = 0
        self.arr_ = None
        self.aligned_ = None

    def gpu_2_cpu(self, read_only = False):
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
            self.gpu_2_cpu(True)
        return super(OpenCLConnector, self).__getstate__()

    def update(self, what_changed = CPU):
        """Updates data ready status.

        Parameters:
            what_changed: what buffer has changed (CPU or GPU).
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

    def _map(self, buf_, OP = opencl.CL_MAP_WRITE):
        self.arr_, event = pyopencl.enqueue_map_buffer(queue=self.device.queue_, buf=buf_, flags=OP, \
            offset=0, shape=self.aligned_.shape, dtype=self.aligned_.dtype, order="C", \
            wait_for=None, is_blocking=False)
        event.wait()
        self.what_changed = 0

    def _unmap(self):
        self.arr_.base.release(queue=self.device.queue_)
        self.arr_ = None
        self.what_changed = 0

    def _write(self, buf_):
        ev = pyopencl.enqueue_copy(self.device.queue_, buf_, self.aligned_, wait_for=None, is_blocking=False)
        ev.wait()
        self.what_changed = 0

    def _read(self, buf_):
        ev = pyopencl.enqueue_copy(self.device.queue_, self.aligned_, buf_, wait_for=None, is_blocking=False)
        ev.wait()
        self.what_changed = 0

    def _buffer(self):
        mf = pyopencl.mem_flags
        if self.device.prefer_mmap:
            buf = pyopencl.Buffer(self.device.context_, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.aligned_)
        else:
            buf = pyopencl.Buffer(self.device.context_, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=self.aligned_.nbytes)
            self.what_changed = CPU
        return buf


class Batch(OpenCLConnector):
    """Batch.

    Attributes:
        batch: numpy array with first dimension as batch.
        batch_: OpenCL buffer mapped to aligned_.
    """
    def __init__(self, device = None, unpickling = 0):
        super(Batch, self).__init__(device=device, unpickling=unpickling)
        self.batch_ = None
        if unpickling:
            return
        self.batch = None

    def initialize(self, device = None):
        if self.batch_:
            return
        if device:
            self.device = device
        if not self.device:
            return
        BLOCK_SIZE = self.device.info.BLOCK_SIZE
        dim1 = self.batch.shape[0]
        dim2 = self.batch.size // self.batch.shape[0]
        if (dim1 % BLOCK_SIZE) or (dim2 % BLOCK_SIZE):
            b = self.batch.reshape([dim1, dim2])
            d1 = dim1
            if d1 % BLOCK_SIZE:
                d1 += BLOCK_SIZE - d1 % BLOCK_SIZE
            d2 = dim2
            if d2 % BLOCK_SIZE:
                d2 += BLOCK_SIZE - d2 % BLOCK_SIZE
            self.aligned_ = filters.aligned_zeros([d1, d2], dtype=self.batch.dtype)
            self.aligned_[0:dim1, 0:dim2] = b[0:dim1, 0:dim2]
            self.batch = self.aligned_[0:dim1, 0:dim2].view().reshape(self.batch.shape)
            assert self.aligned_.__array_interface__["data"][0] == self.batch.__array_interface__["data"][0]
        else:
            self.aligned_ = filters.realign(self.batch)
            self.batch = self.aligned_
        self.batch_ = self._buffer()

    def gpu_2_cpu(self, read_only = False):
        if self.device.prefer_mmap:
            if read_only:
                self._map(self.batch_, opencl.CL_MAP_READ)
                self._unmap()
            else:
                self._map(self.batch_)
        else:
            self._read(self.batch_)

    def cpu_2_gpu(self):
        if self.arr_ != None:
            self._unmap()
        else:
            self._write(self.batch_)


class Vector(OpenCLConnector):
    """Vector.

    Attributes:
        v: numpy array as a vector.
        v_: OpenCL buffer mapped to aligned_.
    """
    def __init__(self, device = None, unpickling = 0):
        super(Vector, self).__init__(device=device, unpickling=unpickling)
        self.v_ = None
        if unpickling:
            return
        self.v = None

    def initialize(self, device = None):
        if self.v_:
            return
        if device:
            self.device = device
        if not self.device:
            return
        BLOCK_SIZE = self.device.info.BLOCK_SIZE
        dim1 = self.v.shape[0]
        dim2 = self.v.size // self.v.shape[0]
        if (dim1 % BLOCK_SIZE) or ((dim2 > 1) and (dim2 % BLOCK_SIZE)):
            b = self.v.reshape([dim1, dim2])
            d1 = dim1
            if d1 % BLOCK_SIZE:
                d1 += BLOCK_SIZE - d1 % BLOCK_SIZE
            d2 = dim2
            if (d2 > 1) and (d2 % BLOCK_SIZE):
                d2 += BLOCK_SIZE - d2 % BLOCK_SIZE
            self.aligned_ = filters.aligned_zeros([d1, d2], dtype=self.v.dtype)
            self.aligned_[0:dim1, 0:dim2] = b[0:dim1, 0:dim2]
            self.v = self.aligned_[0:dim1, 0:dim2].view().reshape(self.v.shape)
            assert self.aligned_.__array_interface__["data"][0] == self.v.__array_interface__["data"][0]
        else:
            self.aligned_ = filters.realign(self.v)
            self.v = self.aligned_
        self.v_ = self._buffer()

    def gpu_2_cpu(self, read_only = False):
        if self.device.prefer_mmap:
            if read_only:
                self._map(self.v_, opencl.CL_MAP_READ)
                self._unmap()
            else:
                self._map(self.v_)
        else:
            self._read(self.v_)

    def cpu_2_gpu(self):
        if self.arr_ != None:
            self._unmap()
        else:
            self._write(self.v_)


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
