"""
Created on Feb 16, 2015

Copyright (c) 2013, Samsung Electronics, Co., Ltd.
"""

import numpy

from veles.memory import Vector
from veles.tests import AcceleratedTest, multi_device, timeout

from veles.znicz import weights_zerofilling


class TestZeroFilling(AcceleratedTest):
    @timeout(100)
    @multi_device()
    def test_zero_filling(self):
        zero_filler = weights_zerofilling.ZeroFiller(self.parent, grouping=2)
        zero_filler.weights = Vector(numpy.ones(shape=(400, 15, 15, 40),
                                                dtype=numpy.float64))

        zero_filler.link_from(self.parent.start_point)

        self.parent.end_point.link_from(zero_filler)

        self.parent.initialize(device=self.device, snapshot=False)

        self.parent.run()
        self.assertIsNone(self.parent.thread_pool.failure)

        zero_filler.weights.map_read()
        zero_filler.mask.map_read()
        self.assertLess(numpy.max(
            zero_filler.mask.mem.ravel() - zero_filler.weights.mem.ravel()),
            0.00001)


if __name__ == "__main__":
    AcceleratedTest.main()
