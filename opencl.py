"""
Created on Mar 21, 2013

OpenCL helper class.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import pyopencl as cl
import time
import numpy
import pickle
import error
import filters


class Device(filters.SmartPickling):
    """OpenCL device helper class.
    
    Attributes:
        guid: "GUID" of the device.
        rating: in [0, 1] interval (1 - fastest, 0.5 - 50% slower than fastest, 0 - unrated).
        dt: time of rating test pass.
        min_dt: minimum time of rating test pass of all tests.
        memsize: "available" size of the memory on the device.
        context_: OpenCL context handle.
    """
    def __init__(self, unpickling = 0, guid = ""):
        super(Device, self).__init__()
        self.context_ = None
        if unpickling:
            return
        self.guid = guid
        self.rating = 0
        self.dt = 604800
        self.min_dt = 86400
        self.memsize = 0


class OpenCL(filters.SmartPickling):
    """OpenCL helper class.

    Attributes:
        devices: dictionary of Device objects where key is device "GUID".
        free_devices: devices marked as free and sorted in rating order.
        events_: dictionary of the queued events.
    """
    def __init__(self, unpickling = 0):
        super(OpenCL, self).__init__(unpickling)
        # reinit all anyway
        self.devices = {}
        self.free_devices = []
        self.events_ = {}
        self._restore()

    def get_free_device(self):
        """Get free device from the list. 

        Raises:
            ErrNotExists.
        """
        if not len(self.free_devices):
            raise error.ErrNotExists()
        return self.free_devices.pop()

    def return_device(self, device):
        """Returns device to the list.

        Raises:
            ErrExists.
        """
        if device in self.free_devices:
            raise error.ErrExists()
        self.free_devices.append(device)
        self.free_devices.sort(key=lambda device: device.rating)

    def _get_device_guid(self, device):
        return (device.get_info(cl.device_info.VENDOR).strip()+"/"+
                device.get_info(cl.device_info.NAME).strip()+"/"+
                str(device.get_info(cl.device_info.VENDOR_ID)))

    def _get_memsize(self, device):
        # MAX_MEM_ALLOC_SIZE usually incorrectly returns only 25% of  the device RAM
        # We will return slightly less amount than the total device RAM
        return device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * 9 // 10

    def _restore(self, do_tests = 1):
        """Initialize an object after restoring from snapshot.
        """
        try:
            fin = open("cache/opencl.pickle", "rb")
            self.devices = pickle.load(fin)
            fin.close()
        except IOError:
            pass
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                guid = self._get_device_guid(device)
                if guid not in self.devices:
                    self.devices[guid] = Device(guid=guid)
                self.devices[guid].memsize = self._get_memsize(device)
                self.devices[guid].context_ = cl.Context([device])
        guids_to_remove = []
        for guid, device in self.devices.items():
            if not device.context_:
                guids_to_remove.append(guid)
        for guid in guids_to_remove:
            del(self.devices[guid])

        if do_tests and self._do_tests():
            print("Saving test results to opencl.pickle...")
            fout = open("cache/opencl.pickle", "wb")
            pickle.dump(self.devices, fout)
            fout.close()
            print("Done")

        for device in self.devices.values():
            self.free_devices.append(device)
        self.free_devices.sort(key=lambda device: device.rating)

    def _do_cpu_test(self):
        """Pure single core CPU test
        """
        b = numpy.copy(self.b.transpose())
        c = numpy.empty_like(self.c)
        numpy.dot(self.a, b, c)
        c[:] += self.bias
        c *= 0.6666
        numpy.tanh(c, c)
        c *= 1.7159
        self.cc = c

    def _do_tests(self):
        """Measure relative device performance.
        """
        for device in self.devices.values():
            if not device.rating:
                break
        else:
            return 0

        self._prepare_tests()

        for device in self.devices.values():
            min_dt = device.min_dt
            break
        print("Test(numpy)...")
        t1 = time.time()
        self._do_cpu_test()
        t2 = time.time()
        dt = t2 - t1
        dt_numpy = dt
        if dt < min_dt:
            min_dt = dt
        print("Done in %.2f seconds" % (dt))
        for guid, device in self.devices.items():
            if device.rating:
                continue
            print("Test(%s)..." % (guid))
            t1 = time.time()
            self._do_test(device.context_)
            t2 = time.time()
            dt = t2 - t1
            device.dt = dt
            if dt < min_dt:
                min_dt = dt
            self.c -= self.cc
            print("Done in %.2f seconds, MSE = %.6f" % (dt, numpy.linalg.norm(self.c) / self.c.size))
        print("\nRating(numpy): %.2f" % (min_dt / dt_numpy))
        for guid, device in self.devices.items():
            rating = min_dt / device.dt
            if device.rating != rating:
                if device.rating:
                    print("UPD Rating(%s): %.2f" % (guid, rating))
                else:
                    print("NEW Rating(%s): %.2f" % (guid, rating))
            else:
                print("Rating(%s): %.2f" % (guid, rating))
            device.rating = rating
            device.min_dt = min_dt

        self._cleanup_after_tests()
        print()
        return 1

    def _prepare_tests(self):
        self.rnd_state = numpy.random.get_state()
        self.a = numpy.random.rand(2048 * 4096).astype(numpy.float32).reshape([2048, 4096])
        self.a -= 0.5
        self.b = numpy.random.rand(4096 * 8192).astype(numpy.float32).reshape([8192, 4096]) # transposed
        self.b -= 0.5
        self.bias = numpy.random.rand(8192).astype(numpy.float32)
        self.bias -= 0.5
        self.c = numpy.empty([2048 * 8192], dtype=numpy.float32).reshape([2048, 8192]) # result

    def _cleanup_after_tests(self):
        del(self.c)
        del(self.bias)
        del(self.b)
        del(self.a)
        numpy.random.set_state(self.rnd_state)
        del(self.rnd_state)

    def _do_test(self, context):
        """Do test for specific context
        """
        queue = cl.CommandQueue(context)

        defines = ("#define BLOCK_SIZE 16\n"
        "#define AB_WIDTH 4096\n"
        "#define B_HEIGHT 8192\n\n")
        fin = open("cl/feed_tanh.cl", "r")
        src = defines + fin.read()
        fin.close()
        fout = open("cache/test.cl", "w")
        fout.write(src)
        fout.close()

        mf = cl.mem_flags
        a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
        b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
        c_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.c)
        bias_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.bias)

        prg = cl.Program(context, src).build()

        krn = cl.Kernel(prg, "FEED_LAYER")
        krn.set_arg(0, a_buf)
        krn.set_arg(1, b_buf)
        krn.set_arg(2, c_buf)
        krn.set_arg(3, bias_buf)

        global_size = [8192, 2048]
        local_size = [16, 16]
        cl.enqueue_nd_range_kernel(queue, krn, global_size, local_size)

        cl.enqueue_copy(queue, self.c, c_buf)

    def add_event(self, event):
        """Adds event to the queue
        """
        self.events_[event] = 1

    def check_for_event(self):
        """Gets event from the queue that is ready
        """
        if not self.events_:
            return None
        for event in self.events_.keys():
            stt = event.command_execution_status
            if stt <= 0:
                del(self.events_[event])
                if stt < 0:
                    raise error.ErrOpenCL(event)
                return event
        return None

    def handle_event(self, event):
        return event.callback(*event.callback_args)
