import logging
import numpy
import unittest


from veles.backends import Device
from veles.config import root
from veles.dummy import DummyWorkflow
from veles.memory import Vector
import veles.opencl_types as opencl_types

from veles.znicz.resizable_all2all import ResizableAll2All


class TestResizableAll2All(unittest.TestCase):

    def setUp(self):
        self.device = Device()

    def tearDown(self):
        pass

    def test_adjust(self):
        dtype = opencl_types.dtypes[root.common.precision_type]
        inp = Vector(numpy.array([[1, 2, 3, 2, 1],
                                  [0, 1, 2, 1, 0],
                                  [0, 1, 0, 1, 0],
                                  [2, 0, 1, 0, 2],
                                  [1, 0, 1, 0, 1]], dtype=dtype))
        ra2a = ResizableAll2All(DummyWorkflow(), output_sample_shape=(3,),
                                weights_stddev=0.05)
        ra2a.input = inp
        ra2a.initialize(device=self.device)
        self.assertEqual(ra2a.output.shape, (5, 3))
        ra2a.run()
        ra2a.output_sample_shape = (6,)
        ra2a.run()
        self.assertEqual(ra2a.output.shape, (5, 6))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
