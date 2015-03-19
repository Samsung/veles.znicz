"""
Created on Oct 30, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import numpy
import os
import time

from veles.config import root
from veles.memory import roundup, Vector
import veles.prng as prng
from veles.dummy import DummyWorkflow
from veles.accelerated_units import TrivialAcceleratedUnit
from veles.tests import AcceleratedTest, assign_backend


class TestMatrixTranspose(AcceleratedTest):
    ABSTRACT = True

    def test_transpose(self):
        prng.get().seed(numpy.frombuffer(b'12345678', dtype=numpy.int32))
        WIDTH = 4096
        HEIGHT = 4096
        dtype = numpy.float32
        a = Vector(numpy.zeros([HEIGHT, WIDTH], dtype=dtype))
        prng.get().fill(a.mem)
        b = Vector(numpy.zeros([WIDTH * 2, HEIGHT], dtype=dtype))

        obj = TrivialAcceleratedUnit(DummyWorkflow())
        obj.initialize(device=self.device)

        a.initialize(self.device)
        b.initialize(self.device)

        gold = a.mem.transpose().copy()

        bs = 16
        obj.sources_["matrix_transpose"] = {
            "BLOCK_SIZE": bs
        }
        obj.build_program(
            {}, os.path.join(root.common.cache_dir, "test"), dtype=dtype)

        krn = obj.get_kernel("transpose")
        krn.set_arg(0, a.devmem)
        krn.set_arg(1, b.devmem)
        ii = numpy.array([WIDTH, HEIGHT], dtype=numpy.int32)
        krn.set_arg(2, ii[0:1])
        krn.set_arg(3, ii[1:2])
        global_size = [roundup(WIDTH, bs), roundup(HEIGHT, bs)]
        local_size = [bs, bs]

        self.device.queue_.execute_kernel(krn, global_size, local_size)
        self.device.queue_.flush()
        self.device.queue_.finish()
        b.map_read()
        max_diff = numpy.fabs(gold - b.mem[:WIDTH, :]).max()
        self.debug("max_diff is %.14f", max_diff)
        self.assertEqual(max_diff, 0.0)
        self.assertEqual(numpy.count_nonzero(b.mem[WIDTH:, :]), 0)
        b.unmap()
        self.device.queue_.flush()
        self.device.queue_.finish()
        N = 3
        t0 = time.time()
        for _ in range(N):
            self.device.queue_.execute_kernel(krn, global_size, local_size)
        self.device.queue_.flush()
        self.device.queue_.finish()
        dt = time.time() - t0
        self.debug("Avg time is %.14f", dt / N)


@assign_backend("ocl")
class OpenCLTestMatrixTranspose(TestMatrixTranspose):
    pass


@assign_backend("cuda")
class CUDATestMatrixTranspose(TestMatrixTranspose):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
