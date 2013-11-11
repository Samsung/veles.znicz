"""
Created on Nov 8, 2013

Will test correctness of OpenCL matrix multiplication.

@author: ajk
"""
import unittest
import opencl
import numpy
import config
import formats
import rnd
import pyopencl as cl


class TestMatrixMultiplication(unittest.TestCase):
    def do_cpu_test(self):
        """Pure single core CPU test
        """
        dtype = (numpy.complex128 if self.a.dtype in (
                    numpy.complex64, numpy.complex128) else numpy.float64)
        a = numpy.empty(self.a.shape, dtype=dtype)
        a[:] = self.a[:]
        bt = self.b.transpose()
        b = numpy.empty(bt.shape, dtype=dtype)
        b[:] = bt[:]
        bias = numpy.empty(self.bias.shape, dtype=dtype)
        bias[:] = self.bias[:]
        c = numpy.empty(self.c[0].shape, dtype=dtype)
        numpy.dot(a, b, c)
        c[:] += bias
        c *= 0.6666
        numpy.tanh(c, c)
        c *= 1.7159
        return c

    def prepare_tests(self, BLOCK_SIZE, dtype=config.dtypes[config.dtype],
                      AB_WIDTH=1371, B_HEIGHT=11735, A_HEIGHT=171):
        self.AB_WIDTH = AB_WIDTH
        self.B_HEIGHT = B_HEIGHT
        self.A_HEIGHT = A_HEIGHT

        self.a = formats.aligned_zeros([self.A_HEIGHT * self.AB_WIDTH],
                                       dtype=dtype)
        rnd.default.fill(self.a, -0.1, 0.1)
        self.a = self.a.reshape([self.A_HEIGHT, self.AB_WIDTH])

        self.b = formats.aligned_zeros([self.B_HEIGHT * self.AB_WIDTH],
                                       dtype=dtype)
        rnd.default.fill(self.b, -0.1, 0.1)
        self.b = self.b.reshape([self.B_HEIGHT, self.AB_WIDTH])

        self.bias = formats.aligned_zeros([self.B_HEIGHT],
                                          dtype=dtype)
        rnd.default.fill(self.bias, -0.1, 0.1)

        self.c = formats.aligned_zeros([2, self.A_HEIGHT, self.B_HEIGHT],
                                       dtype=dtype)

    def cleanup_after_tests(self):
        del(self.c)
        del(self.bias)
        del(self.b)
        del(self.a)
        del(self.A_HEIGHT)
        del(self.B_HEIGHT)
        del(self.AB_WIDTH)

    def do_test(self, device, BLOCK_SIZE):
        """Do test for specific context
        """
        defines = ("%s\n"
        "#define ACTIVATION_TANH\n"
        "#define BLOCK_SIZE %d\n"
        "#define H %d\n"
        "#define Y %d\n"
        "#define BATCH %d\n\n" % (config.cl_defines[config.dtype], BLOCK_SIZE,
                                  self.AB_WIDTH, self.B_HEIGHT, self.A_HEIGHT))
        s = defines
        fin = open("%s/defines.cl" % (config.cl_dir), "r")
        s += fin.read()
        fin.close()
        fin = open("%s/matrix_multiplication.cl" % (config.cl_dir), "r")
        s_mx_mul = fin.read()
        fin.close()
        fin = open("%s/forward.cl" % (config.cl_dir), "r")
        s += fin.read()
        fin.close()
        s = s.replace("MX_MUL", s_mx_mul)
        fout = open("%s/test.cl" % (config.cache_dir), "w")
        fout.write(s)
        fout.close()

        mf = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
        a_buf = cl.Buffer(device.context_, mf, hostbuf=self.a)
        b_buf = cl.Buffer(device.context_, mf, hostbuf=self.b)
        self.c[:] = 0
        c_buf = cl.Buffer(device.context_, mf, hostbuf=self.c)
        bias_buf = cl.Buffer(device.context_, mf, hostbuf=self.bias)

        prg = cl.Program(device.context_, s).build()

        krn = cl.Kernel(prg, "feed_layer")
        krn.set_arg(0, a_buf)
        krn.set_arg(1, b_buf)
        krn.set_arg(2, c_buf)
        krn.set_arg(3, bias_buf)

        global_size = [formats.roundup(self.B_HEIGHT, BLOCK_SIZE),
                       formats.roundup(self.A_HEIGHT, BLOCK_SIZE)]
        local_size = [BLOCK_SIZE, BLOCK_SIZE]

        event = cl.enqueue_nd_range_kernel(device.queue_, krn, global_size,
                                           local_size)
        event.wait()

        event = cl.enqueue_copy(device.queue_, self.c, c_buf,
                                wait_for=None, is_blocking=False)
        event.wait()

    def test(self):
        self.rnd = rnd.Rand()
        self.rnd.seed(numpy.fromfile("/dev/urandom", dtype=numpy.int32,
                                     count=1024))
        cl = opencl.DeviceList()
        device = cl.get_device()
        self.assertNotEqual(device, None, "Could not get OpenCL device.")
        block_size = device.info.BLOCK_SIZE[config.dtype]
        N = 1000
        print("Will test %d matrix multiplications "
              "with BLOCK_SIZE = %d" % (N, block_size))
        for i in range(N):
            AB_WIDTH = self.rnd.randint(1, ((i // 10) + 1) * 100)
            B_HEIGHT = self.rnd.randint(1, ((i // 10) + 1) * 10)
            A_HEIGHT = self.rnd.randint(1, ((i // 10) + 1) * 10)
            print("%d: [%d, %d] * [%d, %d] = [%d, %d]" % (i,
                AB_WIDTH, A_HEIGHT, B_HEIGHT, AB_WIDTH,
                A_HEIGHT, B_HEIGHT))
            self.prepare_tests(block_size, AB_WIDTH=AB_WIDTH,
                               B_HEIGHT=B_HEIGHT, A_HEIGHT=A_HEIGHT)
            c = self.do_cpu_test()
            self.do_test(device, block_size)
            max_diff = numpy.fabs(c.ravel() - self.c[0].ravel()).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            num_nz = numpy.count_nonzero(self.c[1].ravel())
            self.assertEqual(num_nz, 0,
                "Written some values outside of the target array bounds")
            self.cleanup_after_tests()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
