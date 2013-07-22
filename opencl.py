"""
Created on Mar 21, 2013

OpenCL helper classes.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import pyopencl as cl
import time
import numpy
import pickle
import units
import os
import config
import logging


CL_MAP_READ = 1
CL_MAP_WRITE = 2
CL_MAP_WRITE_INVALIDATE_REGION = 4


class Device(units.SmartPickler):
    """OpenCL device helper class.

    Attributes:
        info: DeviceInfo object.
        context_: OpenCL context handle.
        queue_: OpenCL device queue.
        pid: process id.
        prefer_mmap: use map/unmap instead of read/write.
    """
    def __init__(self, info=None, unpickling=0):
        super(Device, self).__init__(unpickling=unpickling)
        self.context_ = None
        self.queue_ = None
        self.pid = os.getpid()
        if unpickling:
            return
        self.info = info
        self.prefer_mmap = True


class DeviceInfo(object):
    """Info about device.

    Attributes:
        guid: "GUID" of the device.
        memsize: "available" size of the memory on the device.
        memalign: best alignment for device buffers.
        rating: in [0, 1] interval (1 - fastest, 0.5 - 50% slower than fastest,
                0 - unrated).
        dt: time of rating test pass.
        min_dt: minimum time of rating test pass of all tests.
        BLOCK_SIZE: best block size for matrix multiplication for the device.
    """
    def __init__(self, guid=""):
        self.guid = guid
        self.memsize = 0
        self.memalign = 32
        self.rating = {}
        for dtype in config.dtypes.keys():
            self.rating[dtype] = 0.0
        self.dt = {}
        for dtype in config.dtypes.keys():
            self.dt[dtype] = 86400
        self.min_dt = {}
        for dtype in config.dtypes.keys():
            self.min_dt[dtype] = 86400
        self.BLOCK_SIZE = {}
        for dtype in config.dtypes.keys():
            self.BLOCK_SIZE[dtype] = 8


class DeviceList(units.SmartPickler):
    """Contains list of devices sorted by rating.

    Attributes:
        device_infos: dictionary of device infos by guid.
        devices_available: list of devices available at the time of run
                           sorted by ratings.
        devices_in_use: list of device objects currently in use.
        last_index: index of the last device returned by get_device()
                    in the devices_available list.
    """
    def __init__(self, unpickling=0):
        super(DeviceList, self).__init__(unpickling=unpickling)
        self.device_infos = {}
        try:
            fin = open("%s/device_infos.pickle" % (config.cache_dir,), "rb")
            self.device_infos = pickle.load(fin)
            fin.close()
        except IOError:
            pass

        self.devices_available = []
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices()
            if __debug__:
                print(devices)
            context = cl.Context(devices)
            for device in devices:
                guid = self._get_device_guid(device)
                if guid not in self.device_infos.keys():
                    info = DeviceInfo(guid=guid)
                    self.device_infos[guid] = info
                info = self.device_infos[guid]
                info.memsize = self._get_memsize(device)
                info.memalign = device.get_info(
                    cl.device_info.MEM_BASE_ADDR_ALIGN)
                dev = Device(info=info)
                dev.context_ = context
                dev.queue_ = cl.CommandQueue(context,
                    properties=cl.command_queue_properties.\
                        OUT_OF_ORDER_EXEC_MODE_ENABLE)
                self.devices_available.append(dev)

        if self._do_tests():
            logging.info("Saving test results to %s/device_infos.pickle..." % (
                                                        config.cache_dir,))
            fout = open("%s/device_infos.pickle" % (config.cache_dir,), "wb")
            pickle.dump(self.device_infos, fout)
            fout.close()
            logging.info("Done")

        self.devices_available.sort(
            key=lambda device: device.info.rating[config.dtype],
            reverse=True)
        # leave only one context
        context = self.devices_available[0].context_
        n = len(self.devices_available)
        for i in range(n - 1, 0, -1):
            if self.devices_available[i].context_ != context:
                self.devices_available.pop(i)
        logging.info("Selected single context with the following devices "
              "(guid: dtype, rating, BLOCK_SIZE, memalign):")
        for device in self.devices_available:
            for dtype in device.info.rating.keys():
                logging.info("%s: %s, %.4f, %s, %d" % (device.info.guid, dtype,
                    device.info.rating[dtype], device.info.BLOCK_SIZE[dtype],
                    device.info.memalign))

        if not unpickling:
            self.devices_in_use = []
        self.last_index = 0
        for self.last_index in range(0, len(self.devices_in_use)):
            self.devices_in_use[self.last_index].__dict__.\
            update(self.devices_available[self.last_index %
                   len(self.devices_available)].__dict__)

    def get_device(self):
        """Gets device from the available list.
        """
        self.last_index += 1
        dev = Device()
        dev.__dict__.update(self.devices_available[self.last_index %
                            len(self.devices_available)].__dict__)
        self.devices_in_use.append(dev)
        return dev

    def return_device(self, dev):
        """Returns device to the available list.
        """
        idx = 0
        for i in range(0, len(self.devices_available)):
            if self.devices_available[i].context_ == dev.context_:
                idx = i
                break
        else:
            return
        self.devices_available.insert((self.last_index - 1) %
                                      len(self.devices_available),
                                      self.devices_available.pop(idx))
        self.devices_in_use.remove(dev)

    def _get_device_guid(self, device):
        return ("%s/%s/%s" % (device.get_info(cl.device_info.VENDOR).strip(),
                device.get_info(cl.device_info.NAME).strip(),
                str(device.get_info(cl.device_info.VENDOR_ID))))

    def _get_memsize(self, device):
        # MAX_MEM_ALLOC_SIZE usually incorrectly returns only 25%
        # of the device RAM, we will return slightly less amount
        # than the total device RAM.
        return device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * 9 // 10

    def _do_cpu_test(self):
        """Pure single core CPU test
        """
        a = numpy.empty(self.a.shape, dtype=numpy.float64)
        a[:] = self.a[:]
        bt = self.b.transpose()
        b = numpy.empty(bt.shape, dtype=numpy.float64)
        b[:] = bt[:]
        bias = numpy.empty(self.bias.shape, dtype=numpy.float64)
        bias[:] = self.bias[:]
        c = numpy.empty(self.c.shape, dtype=numpy.float64)
        t1 = time.time()
        numpy.dot(a, b, c)
        c[:] += bias
        c *= 0.6666
        numpy.tanh(c, c)
        c *= 1.7159
        dt = time.time() - t1
        self.cc = c
        return dt

    def _do_tests(self):
        """Measure relative device performance.
        """
        for device in self.devices_available:
            for dtype in config.dtypes.keys():
                if not device.info.rating[dtype]:
                    break
            else:
                continue
            break
        else:
            return 0

        min_dt = {}
        for dtype in config.dtypes.keys():
            min_dt[dtype] = 86400
        dt_numpy = 86400
        for info in self.device_infos.values():
            for dtype in info.min_dt.keys():
                min_dt[dtype] = info.min_dt[dtype]
            break

        for device in self.devices_available:
            for dtype in config.dtypes.keys():
                if not device.info.rating[dtype]:
                    break
            else:
                continue
            for dtype in device.info.dt.keys():
                device.info.dt[dtype] = 86400
            for BLOCK_SIZE in range(32, 3, -1):
                for dtype in config.dtypes.keys():
                    try:
                        self._prepare_tests(BLOCK_SIZE, dtype)
                        b_numpy = False
                        if BLOCK_SIZE == 32 and dt_numpy == 86400:
                            b_numpy = True
                            logging.info("Numpy double precision...")
                            dt = self._do_cpu_test()
                            logging.info("Done in %.2f seconds" % (dt,))
                            if dt < dt_numpy:
                                dt_numpy = dt
                        if dt_numpy < min_dt[dtype]:
                            min_dt[dtype] = dt_numpy
                        logging.info("Testing %s with BLOCK_SIZE = %d "
                              "and dtype = %s" % \
                              (device.info.guid, BLOCK_SIZE, dtype))
                        dt = self._do_test(device, BLOCK_SIZE, dtype, 7)
                        if dt < device.info.dt[dtype]:
                            device.info.dt[dtype] = dt
                            device.info.BLOCK_SIZE[dtype] = BLOCK_SIZE
                        if dt < min_dt[dtype]:
                            min_dt[dtype] = dt
                        if b_numpy:
                            c = self.cc.copy()
                            c -= self.c
                            numpy.abs(c, c)
                            logging.info("Avg is %.2f seconds, MSE = %.6f, "
                                  "max_diff = %.6f" %
                                  (dt, numpy.linalg.norm(c) / c.size, c.max()))
                        else:
                            logging.info("Avg is %.2f seconds" % (dt,))
                        self._cleanup_after_tests()
                    except (cl.LogicError, cl.RuntimeError, cl.MemoryError):
                        logging.info("Program compilation or run failed for "
                              "BLOCK_SIZE = %d and dtype = %s" % (BLOCK_SIZE,
                                                                  dtype))
                        self._cleanup_after_tests()
                        #raise

        logging.info("\nRating(numpy double precision): %.4f" % \
              (min_dt[config.dtype] / dt_numpy))
        for info in self.device_infos.values():
            for dtype in config.dtypes.keys():
                logging.info("")
                logging.info(dtype)
                rating = min_dt[dtype] / info.dt[dtype]
                if info.rating[dtype] != rating:
                    if info.rating[dtype]:
                        logging.info("UPD Rating(%s): %.4f" % (info.guid,
                                                               rating))
                    else:
                        logging.info("NEW Rating(%s): %.4f" % (info.guid,
                                                               rating))
                else:
                    logging.info("Rating(%s): %.4f" % (info.guid, rating))
                info.rating[dtype] = rating
                info.min_dt[dtype] = min_dt[dtype]
        logging.info("")
        return 1

    def _prepare_tests(self, BLOCK_SIZE, dtype):
        self.AB_WIDTH = 128 * 1024
        self.B_HEIGHT = 256
        self.A_HEIGHT = 512
        if self.AB_WIDTH % BLOCK_SIZE:
            self.AB_WIDTH += BLOCK_SIZE - self.AB_WIDTH % BLOCK_SIZE
        if self.B_HEIGHT % BLOCK_SIZE:
            self.B_HEIGHT += BLOCK_SIZE - self.B_HEIGHT % BLOCK_SIZE
        if self.A_HEIGHT % BLOCK_SIZE:
            self.A_HEIGHT += BLOCK_SIZE - self.A_HEIGHT % BLOCK_SIZE
        logging.info("Matricies are: [%d, %d] * [%d, %d] = [%d, %d]" % (
            self.AB_WIDTH, self.A_HEIGHT, self.B_HEIGHT, self.AB_WIDTH,
            self.A_HEIGHT, self.B_HEIGHT))
        self.rnd_state = numpy.random.get_state()

        self.a = units.aligned_zeros([self.A_HEIGHT * self.AB_WIDTH],
                                     dtype=config.dtypes[dtype])
        self.a[:] = numpy.random.rand(self.a.size)
        self.a -= 0.5
        self.a = self.a.reshape([self.A_HEIGHT, self.AB_WIDTH])

        self.b = units.aligned_zeros([self.B_HEIGHT * self.AB_WIDTH],
                                     dtype=config.dtypes[dtype])
        self.b[:] = numpy.random.rand(self.b.size)
        self.b -= 0.5
        self.b = self.b.reshape([self.B_HEIGHT, self.AB_WIDTH])

        self.bias = units.aligned_zeros([self.B_HEIGHT],
                                        dtype=config.dtypes[dtype])
        self.bias[:] = numpy.random.rand(self.bias.size)
        self.bias -= 0.5

        self.c = units.aligned_zeros([self.A_HEIGHT, self.B_HEIGHT],
                                     dtype=config.dtypes[dtype])

    def _cleanup_after_tests(self):
        if "cc" in self.__dict__:
            del(self.cc)
        del(self.c)
        del(self.bias)
        del(self.b)
        del(self.a)
        numpy.random.set_state(self.rnd_state)
        del(self.rnd_state)
        del(self.A_HEIGHT)
        del(self.B_HEIGHT)
        del(self.AB_WIDTH)

    def _do_test(self, device, BLOCK_SIZE, dtype, iters):
        """Do test for specific context
        """
        defines = ("%s\n"
        "#define ACTIVATION_TANH\n"
        "#define BLOCK_SIZE %d\n"
        "#define H %d\n"
        "#define Y %d\n"
        "#define BATCH %d\n\n" % (config.cl_defines[dtype], BLOCK_SIZE,
                                  self.AB_WIDTH, self.B_HEIGHT, self.A_HEIGHT))
        fin = open("%s/matrix_multiplication.cl" % (config.cl_dir,), "r")
        s_mx_mul = fin.read()
        fin.close()
        fin = open("%s/forward.cl" % (config.cl_dir,), "r")
        s = defines + fin.read()
        fin.close()
        s = s.replace("MX_MUL", s_mx_mul)
        fout = open("%s/test.cl" % (config.cache_dir,), "w")
        fout.write(s)
        fout.close()

        mf = cl.mem_flags
        a_buf = cl.Buffer(device.context_, mf.READ_ONLY | mf.USE_HOST_PTR,
                          hostbuf=self.a)
        b_buf = cl.Buffer(device.context_, mf.READ_ONLY | mf.USE_HOST_PTR,
                          hostbuf=self.b)
        self.c[:] = 0
        c_buf = cl.Buffer(device.context_, mf.WRITE_ONLY | mf.USE_HOST_PTR,
                          hostbuf=self.c)
        bias_buf = cl.Buffer(device.context_, mf.READ_ONLY | mf.USE_HOST_PTR,
                             hostbuf=self.bias)

        prg = cl.Program(device.context_, s).build()

        krn = cl.Kernel(prg, "FEED_LAYER")
        krn.set_arg(0, a_buf)
        krn.set_arg(1, b_buf)
        krn.set_arg(2, c_buf)
        krn.set_arg(3, bias_buf)

        global_size = [self.B_HEIGHT, self.A_HEIGHT]
        local_size = [BLOCK_SIZE, BLOCK_SIZE]
        t1 = time.time()
        # Will skip the first iteration
        for i in range(0, iters + 1):
            if i == 1:
                t1 = time.time()
            event = cl.enqueue_nd_range_kernel(device.queue_, krn, global_size,
                                               local_size)
            event.wait()
        dt = time.time() - t1
        # Get results back
        arr, event = cl.enqueue_map_buffer(queue=device.queue_, buf=c_buf,
            flags=CL_MAP_READ, offset=0, shape=self.c.shape,
            dtype=self.c.dtype, order="C", wait_for=None, is_blocking=False)
        event.wait()
        arr.base.release(queue=device.queue_, wait_for=None)
        return dt / iters
