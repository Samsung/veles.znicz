"""
Created on Apr 15, 2013

Data formats for connectors.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import units
import pyopencl
import os
import config
import error
import logging


CPU = 1
GPU = 2


def assert_addr(a, b):
    """Raises an exception if addresses of the supplied arrays differ.
    """
    if a.__array_interface__["data"][0] != b.__array_interface__["data"][0]:
        raise error.ErrBadFormat("Addresses of the arrays are not equal.")


def ravel(a):
    """numpy.ravel() with address check.
    """
    b = a.ravel()
    assert_addr(a, b)
    return b


def reshape(a, shape):
    """numpy.reshape() with address check.
    """
    b = a.reshape(shape)
    assert_addr(a, b)
    return b


def real_normalize(a):
    """Normalizes real array to [-1, 1] in-place.
    """
    a -= a.min()
    m = a.max()
    if m:
        a /= m
        a *= 2.0
        a -= 1.0


def complex_normalize(a):
    """Normalizes complex array to unit-circle in-place.
    """
    a -= numpy.average(a)
    m = numpy.sqrt((a.real * a.real + a.imag * a.imag).max())
    if m:
        a /= m


def normalize(a):
    if a.dtype in (numpy.complex64, numpy.complex128):
        return complex_normalize(a)
    return real_normalize(a)


def normalize_pointwise(a):
    """Normalizes dataset pointwise.
    """
    IMul = numpy.zeros_like(a[0])
    IAdd = numpy.zeros_like(a[0])

    mins = numpy.min(a, 0)
    maxs = numpy.max(a, 0)
    ds = maxs - mins
    zs = numpy.nonzero(ds)

    IMul[zs] = 2.0
    IMul[zs] /= ds[zs]

    mins *= IMul
    IAdd[zs] = -1.0
    IAdd[zs] -= mins[zs]

    logging.info("%f %f %f %f" % (IMul.min(), IMul.max(),
                                  IAdd.min(), IAdd.max()))

    return (IMul, IAdd)


def realign(arr, boundary=4096):
    """Reallocates array to become PAGE-aligned as required for
        clEnqueueMapBuffer().
    """
    if arr == None:
        return None
    address = arr.__array_interface__["data"][0]
    if address % boundary == 0:
        return arr
    N = numpy.prod(arr.shape)
    d = arr.dtype
    tmp = numpy.empty(N * d.itemsize + boundary, dtype=numpy.uint8)
    address = tmp.__array_interface__["data"][0]
    offset = (boundary - address % boundary) % boundary
    a = tmp[offset:offset + N * d.itemsize].view(dtype=d)
    b = a.reshape(arr.shape, order="C")
    assert_addr(a, b)
    b[:] = arr[:]
    return b


def aligned_zeros(shape, boundary=4096, dtype=numpy.float32):
    """Allocates PAGE-aligned array required for clEnqueueMapBuffer().
    """
    N = numpy.prod(shape)
    d = numpy.dtype(dtype)
    tmp = numpy.zeros(N * d.itemsize + boundary, dtype=numpy.uint8)
    address = tmp.__array_interface__["data"][0]
    offset = (boundary - address % boundary) % boundary
    a = tmp[offset:offset + N * d.itemsize].view(dtype=d)
    b = a.reshape(shape, order="C")
    assert_addr(a, b)
    return b


class Vector(units.Connector):
    """Container class for numpy array backed by OpenCL buffer.

    Attributes:
        device: OpenCL device.
        what_changed: what buffer has changed (CPU or GPU).
        v: numpy array.
        aligned_: numpy array, aligned to BLOCK_SIZE
                  for OpenCL matrix multiplication.
        v_: OpenCL buffer mapped to aligned_.
        supposed_maxvle: supposed maximum element value.

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
        self.what_changed = CPU
        self.v = None
        self.supposed_maxvle = 1.0

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
        if self.v == None or self.v_ != None:
            return
        if device != None:
            self.device = device
        if self.device == None:
            return
        if (self.v.dtype in config.convert_map.keys() and
            config.convert_map[self.v.dtype] in [
                config.dtypes[config.dtype], config.dtypes[config.c_dtype]]):
            self.v = self.v.astype(config.convert_map[self.v.dtype])
        BLOCK_SIZE = self.device.info.BLOCK_SIZE[config.c_dtype]
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
            self.aligned_ = aligned_zeros([d1, d2],
                boundary=self.device.info.memalign, dtype=self.v.dtype)
            self.aligned_[0:dim1, 0:dim2] = b[0:dim1, 0:dim2]
            self.v = self.aligned_[0:dim1, 0:dim2].view().reshape(self.v.shape)
            assert_addr(self.aligned_, self.v)
        else:
            self.aligned_ = realign(self.v, self.device.info.memalign)
            self.v = self.aligned_
        mf = pyopencl.mem_flags
        self.v_ = pyopencl.Buffer(self.device.context_,
                mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.aligned_)
        self.what_changed = 0

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
        self.what_changed = CPU
        self.v = None
        self.aligned_ = None
        self.v_ = None
