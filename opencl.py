"""
Created on Mar 21, 2013

OpenCL helper class.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import pyopencl as cl
import time
import numpy


class DeviceInfo(object):
    """Some useful preprocessed OpenCL device info.
    
    Attributes:
        memsize: total memory size available for the program.
        rating: relative performance rating of the device.
    """
    def __init__(self, device):
        super(DeviceInfo, self).__init__()
        # MAX_MEM_ALLOC_SIZE usually incorrectly returns only 25% of  the device RAM
        # We will return slightly less amount than the total device RAM
        self.memsize = device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * 9 // 10;
        self.rating = 1.0


class OpenCL(object):
    """OpenCL helper class.

    TODO(a.kazantsev): restrict devices available.

    Attributes:
        platforms: list of OpenCL platforms.
        devices: dictionary of devices lists per platform
        infos: dictionary of infos per device
        contexts: dictionary of contexts per platform
    """
    def __init__(self):
        super(OpenCL, self).__init__()
        self.platforms = []
        self.devices = {}
        self.infos = {}
        self.contexts = {}
        self.restore()

    def __getstate__(self):
        """Do not pickle anything.
        """
        return {}

    def _create_contexts(self):
        """Creates OpenCL contexts one per device.
        """
        for device in self.infos.keys():
            self.contexts[device] = cl.Context([device])

    def restore(self, do_tests = 1):
        """Initialize an object after restoring from snapshot.
        """
        self.platforms = cl.get_platforms()
        for platform in self.platforms:
            self.devices[platform] = platform.get_devices()
            for device in self.devices[platform]:
                self.infos[device] = DeviceInfo(device)
        self._create_contexts()
        if do_tests:
            self.do_tests()

    def do_cpu_test(self):
        """Pure single core CPU test
        """
        b = numpy.copy(self.b.transpose())
        numpy.dot(self.a, b, self.c)
        self.c[:] += self.bias
        self.c *= 0.6666
        numpy.tanh(self.c, self.c)
        self.c *= 1.7159

    def do_tests(self):
        """Measure relative device performance.
        """
        self.prepare_tests()
        
        times = {}
        min_dt = 3600.0
        print("Test pure CPU implementation...")
        t1 = time.time()
        self.do_cpu_test()
        t2 = time.time()
        dt = t2 - t1
        times["CPU"] = dt
        if dt < min_dt:
            min_dt = dt
            print("(min, max, sum, [151, 71], [1735, 3111]): (%f, %f, %f, %f, %f)" %
                  (self.c.min(), self.c.max(), self.c.sum(), self.c[151, 71], self.c[1735, 3111]))
        print("Done in %.2f seconds" % (dt))
        for device, context in self.contexts.items():
            print("Test(%s)..." % (device))
            t1 = time.time()
            self.do_test(context)
            t2 = time.time()
            dt = t2 - t1
            times[device] = dt
            if dt < min_dt:
                min_dt = dt
            print("(min, max, sum, [151, 71], [1735, 3111]): (%f, %f, %f, %f, %f)" %
                  (self.c.min(), self.c.max(), self.c.sum(), self.c[151, 71], self.c[1735, 3111]))
            print("Done in %.2f seconds" % (dt))
        for device, dt in times.items():
            if device == "CPU":
                print("Pure CPU rating: %.2f" % (min_dt / dt))
            else:
                self.infos[device].rating = min_dt / dt
                print("Rating(%s): %.2f" % (device, self.infos[device].rating))

        self.cleanup_after_tests()

    def prepare_tests(self):
        self.a = numpy.random.rand(2048 * 4096).astype(numpy.float32).reshape([2048, 4096])
        self.a -= 0.5
        self.b = numpy.random.rand(4096 * 8192).astype(numpy.float32).reshape([8192, 4096]) # transposed
        self.b -= 0.5
        self.bias = numpy.random.rand(8192).astype(numpy.float32)
        self.bias -= 0.5
        self.c = numpy.empty([2048 * 8192], dtype=numpy.float32).reshape([2048, 8192]) # result

    def cleanup_after_tests(self):
        del(self.c)
        del(self.bias)
        del(self.b)
        del(self.a)

    def do_test(self, context):
        """Do test for specific context
        """
        queue = cl.CommandQueue(context)

        defines = ("#define BLOCK_SIZE 16\n"
        "#define AB_WIDTH 4096\n"
        "#define B_HEIGHT 8192\n\n")
        fin = open("feed_tanh.cl", "r")
        src = defines + fin.read()
        fin.close()
        fout = open("test.cl", "w")
        fout.write(src)
        fout.close()

        mf = cl.mem_flags
        a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
        b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
        self.c[:, :] = 0
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
