"""
Created on Nov 19, 2013

Test for OpenCL kernel err_h_reduce() in gradient_descent_conv.cl

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import unittest
import formats
import opencl
import numpy
import config
import pyopencl
import units
import rnd


class TestGDConvErrH(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        del self.device
        units.pool.shutdown()

    def build_program(self, dtype, batch_size, sx, sy, n_channels,
                      kx, ky, n_kernels,
                      err_h_tmp_, err_h, reduce_size,
                      apply_gradient=True, weights_transposed=False,
                      store_gradient=False):
        defines = ("%s\n"
                   "%s\n"
                   "%s\n"
                   "%s\n"
                   "#define BLOCK_SIZE %d\n"
                   "#define BATCH %d\n"
                   "#define SX %d\n"
                   "#define SY %d\n"
                   "#define N_CHANNELS %d\n"
                   "#define KX %d\n"
                   "#define KY %d\n"
                   "#define N_KERNELS %d\n"
                   "#define REDUCE_SIZE %d\n" % (
                   "#define APPLY_GRADIENT"
                   if apply_gradient else "",
                   "#define WEIGHTS_TRANSPOSED"
                   if weights_transposed else "",
                   "#define STORE_GRADIENT"
                   if store_gradient else "",
                   config.cl_defines[dtype],
                   self.device.info.BLOCK_SIZE[dtype],
                   batch_size, sx, sy, n_channels, kx, ky,
                   n_kernels, reduce_size))
        bld = units.OpenCLUnit(self.device)
        bld.cl_sources_["%s/gradient_descent_conv.cl" % (config.cl_dir)] = ""
        bld.build_program(defines)

        krn_err_h_tmp_ = pyopencl.Kernel(bld.prg_, "err_h_reduce")
        krn_err_h_tmp_.set_arg(0, err_h_tmp_.v_)
        krn_err_h_tmp_.set_arg(1, err_h.v_)

        return krn_err_h_tmp_

    def testFixed(self):
        """Test with fixed precomputed result.
        """
        dtype = numpy.float32

        err_h_tmp_ = formats.Vector()
        err_h_tmp_.v = numpy.array([[1, 2, 3, 4, 5, 6, 7],
                                    [-1, -2, -3, -4, -5, -6, -7],
                                    [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4]],
                                   dtype=dtype)
        err_h = formats.Vector()
        err_h.v = numpy.zeros(err_h_tmp_.v.shape[0] * 2, dtype=dtype)

        err_h_tmp_.initialize(self.device)
        err_h.initialize(self.device)

        t = numpy.array([28, -28, 0.4], dtype=dtype)

        for reduce_size in range(1, 15):
            krn_err_h_tmp_ = self.build_program("float", 1, 7, 1, 1, 7, 1, 1,
                                                err_h_tmp_, err_h, reduce_size)
            local_size = [reduce_size]
            global_size = [err_h_tmp_.v.shape[0] * reduce_size]
            ev = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                    krn_err_h_tmp_, global_size, local_size)
            ev.wait()
            err_h.map_write()

            max_diff = numpy.fabs(t - err_h.v[:t.size]).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(numpy.count_nonzero(err_h.v[t.size:]), 0,
                             "Overflow detected")

            err_h.v[:] = 0
            err_h.unmap()

    def testRandom(self):
        """Test with random data vs numpy.
        """
        dtype = config.dtypes[config.c_dtype]

        err_h_tmp_ = formats.Vector()
        err_h_tmp_.v = numpy.zeros([7770, 351], dtype=dtype)
        rnd.default.fill(err_h_tmp_.v)
        err_h = formats.Vector()
        err_h.v = numpy.zeros(err_h_tmp_.v.shape[0] * 2, dtype=dtype)

        err_h_tmp_.initialize(self.device)
        err_h.initialize(self.device)

        t = numpy.sum(err_h_tmp_.v, axis=1)

        for reduce_size in range(1, 400, 50):
            krn_err_h_tmp_ = self.build_program(config.c_dtype, 1, 117, 3, 1,
                                117, 3, 1, err_h_tmp_, err_h, reduce_size)
            local_size = [reduce_size]
            global_size = [err_h_tmp_.v.shape[0] * reduce_size]
            ev = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                    krn_err_h_tmp_, global_size, local_size)
            ev.wait()
            err_h.map_write()

            max_diff = numpy.fabs(t - err_h.v[:t.size]).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(numpy.count_nonzero(err_h.v[t.size:]), 0,
                             "Overflow detected")

            err_h.v[:] = 0
            err_h.unmap()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
