"""
Created on Apr 15, 2013

Data formats for connectors.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import pyopencl
import os
import config
import error


CPU = 1
GPU = 2


class Vector(units.Connector):
    """Container class for numpy array backed by OpenCL buffer.

    Attributes:
        device: OpenCL device.
        what_changed: what buffer has changed (CPU or GPU).
        v: numpy array.
        aligned_: numpy array, aligned to BLOCK_SIZE
                  for OpenCL matrix multiplication.
        v_: OpenCL buffer mapped to aligned_.

    Example of how to use:
        1. Construct an object:
            a = formats.Vector()
        2. Connect units be data:
            u2.b = u1.a
        3. Initialize numpy array:
            a.v = numpy.zeros(...)
        4. Initialize an object with device:
            a.initialize(device)
        5. Set OpenCL buffer as kernel parameter:
            krn.set_arg(0, a.v_)
    """
    def __init__(self, device=None):
        super(Vector, self).__init__()
        self.device = device
        self.what_changed = 0
        self.v = None

    def init_unpickled(self):
        super(Vector, self).init_unpickled()
        self.aligned_ = None
        self.v_ = None

    def __getstate__(self):
        """Get data from OpenCL device before pickling.
        """
        if (self.device and (self.what_changed & GPU) and
            self.device.pid_ == os.getpid()):
            self.gpu_2_cpu()
        return super(Vector, self).__getstate__()

    def initialize(self, device=None):
        if self.v_:
            return
        if device:
            self.device = device
        if not self.device:
            return
        if self.v.dtype in config.dtypes.values() and \
           config.dtypes[config.dtype] != self.v.dtype:
            self.v = self.v.astype(config.dtype)
        BLOCK_SIZE = self.device.info.BLOCK_SIZE[config.dtype]
        dim1 = self.v.shape[0]
        n_dims = len(self.v.shape)
        dim2 = self.v.size // self.v.shape[0]
        if self.v.dtype in config.dtypes.values() and \
           ((dim1 % BLOCK_SIZE) or ((n_dims > 1) and (dim2 % BLOCK_SIZE))):
            b = self.v.reshape([dim1, dim2])
            d1 = dim1
            if d1 % BLOCK_SIZE:
                d1 += BLOCK_SIZE - d1 % BLOCK_SIZE
            d2 = dim2
            if (n_dims > 1) and (d2 % BLOCK_SIZE):
                d2 += BLOCK_SIZE - d2 % BLOCK_SIZE
            self.aligned_ = units.aligned_zeros([d1, d2],
                boundary=self.device.info.memalign, dtype=self.v.dtype)
            self.aligned_[0:dim1, 0:dim2] = b[0:dim1, 0:dim2]
            self.v = self.aligned_[0:dim1, 0:dim2].view().reshape(self.v.shape)
            if (self.aligned_.__array_interface__["data"][0] !=
                self.v.__array_interface__["data"][0]):
                raise error.VelesException(
                    "Address after ndarray.view() differs from original one.")
        else:
            self.aligned_ = units.realign(self.v, self.device.info.memalign)
            self.v = self.aligned_
        mf = pyopencl.mem_flags
        self.v_ = pyopencl.Buffer(self.device.context_,
                mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.aligned_)

    def update(self, what_changed=CPU):
        """Sets where the data has been changed (on CPU or GPU).

        Parameters:
            what_changed: what buffer has changed (CPU or GPU).
        """
        self.what_changed = what_changed
        super(Vector, self).update()

    def sync(self, what_will_use=CPU):
        """Gets data from GPU to CPU or vice versa if neccessary.
        """
        if not self.device:
            return
        if (what_will_use & CPU) and (self.what_changed & GPU):
            self.gpu_2_cpu()
            return
        if (what_will_use & GPU) and (self.what_changed & CPU):
            self.cpu_2_gpu()
            return

    def gpu_2_cpu(self):
        ev = pyopencl.enqueue_copy(self.device.queue_, self.aligned_, self.v_,
                                   wait_for=None, is_blocking=False)
        ev.wait()
        self.what_changed = 0

    def cpu_2_gpu(self):
        ev = pyopencl.enqueue_copy(self.device.queue_, self.v_, self.aligned_,
                                   wait_for=None, is_blocking=False)
        ev.wait()
        self.what_changed = 0

    def __len__(self):
        """To enable [] operator.
        """
        return self.v.size

    def __getitem__(self, key):
        """To enable [] operator.
        """
        return self.v[key]

    def __setitem__(self, key, value):
        """To enable [] operator.
        """
        self.v[key] = value

    def reset(self):
        """Sets buffers to None
        """
        self.what_changed = 0
        self.v = None
        self.aligned_ = None
        self.v_ = None
