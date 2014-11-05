"""
Created on Jul 8, 2014

Will test correctness of Loader.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest
from zope.interface import implementer

from veles.config import root
import veles.opencl as opencl
import veles.prng as rnd
from veles.znicz.loader import IFullBatchLoader, FullBatchLoaderMSE
from veles.dummy import DummyWorkflow


@implementer(IFullBatchLoader)
class Loader(FullBatchLoaderMSE):
    """Loads MNIST dataset.
    """
    def load_data(self):
        """Here we will load MNIST data.
        """
        N = 71599
        self.original_labels.mem = numpy.zeros([N], dtype=numpy.int32)
        self.original_data.mem = numpy.zeros([N, 28, 28],
                                             dtype=numpy.float32)
        # Will use different dtype for target
        self.original_targets.mem = numpy.zeros([N, 3, 3, 3],
                                                dtype=numpy.int16)

        self.original_labels.mem[:] = rnd.get().randint(
            0, 1000, self.original_labels.size)
        rnd.get().fill(self.original_data.mem, -100, 100)
        self.original_targets.plain[:] = rnd.get().randint(
            27, 1735, self.original_targets.size)

        self.class_lengths[0] = 0
        self.class_lengths[1] = 9737
        self.class_lengths[2] = N - self.class_lengths[1]


class TestFullBatchLoader(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def test_random(self):
        results = []
        for device in (self.device, None):
            for on_device in (True, False):
                results.append(self._test_random(device, on_device))
        for i in range(1, len(results)):
            for j in range(len(results[0])):
                max_diff = numpy.fabs(results[i][j] - results[0][j]).max()
                self.assertEqual(max_diff, 0)

    def _test_random(self, device, on_device, N=1000):
        rnd.get().seed(123)
        unit = Loader(DummyWorkflow(), on_device=on_device,
                      prng=rnd.get())
        unit.initialize(device)
        res_data = numpy.zeros([N] + list(unit.minibatch_data.shape),
                               dtype=unit.minibatch_data.dtype)
        res_labels = numpy.zeros([N] + list(unit.minibatch_labels.shape),
                                 dtype=unit.minibatch_labels.dtype)
        res_target = numpy.zeros([N] + list(unit.minibatch_targets.shape),
                                 dtype=unit.minibatch_targets.dtype)
        for i in range(N):
            unit.run()
            unit.minibatch_data.map_read()
            unit.minibatch_labels.map_read()
            unit.minibatch_targets.map_read()
            res_data[i] = unit.minibatch_data.mem
            res_labels[i] = unit.minibatch_labels.mem
            res_target[i] = unit.minibatch_targets.mem
        return (res_data, res_labels, res_target)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
