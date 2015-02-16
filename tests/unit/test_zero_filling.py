import logging
import numpy
import unittest


import veles.backends as opencl
from veles.dummy import DummyWorkflow
from veles.memory import Vector

from veles.znicz import weights_zerofilling


class TestZeroFilling(unittest.TestCase):

    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        pass

    def test_zero_filling(self):

        device = opencl.Device()
        workflow = DummyWorkflow()

        zero_filler = weights_zerofilling.ZeroFiller(
            workflow, n_kernels=400, ky=15, kx=15, n_channels=40, grouping=2)
        zero_filler.weights = Vector(numpy.ones(shape=(400, 15, 15, 40),
                                                dtype=numpy.float64))

        zero_filler.link_from(workflow.start_point)

        workflow.end_point.link_from(zero_filler)

        workflow.initialize(device=device)

        workflow.run()

        zero_filler.weights.map_read()
        zero_filler.mask.map_read()
        logging.info("Error: %f" % numpy.average(zero_filler.mask.mem -
                                                 zero_filler.weights.mem))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running ZeroFilling tests")
    unittest.main()
