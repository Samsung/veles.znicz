"""
Created on February ??, 2015

A unit test for ResizableAll2All - recursive net building block.
"""


import numpy

from veles.config import root
from veles.memory import Vector
import veles.opencl_types as opencl_types
from veles.tests import AcceleratedTest, assign_backend

from veles.znicz.resizable_all2all import ResizableAll2All


class TestResizableAll2All(AcceleratedTest):
    ABSTRACT = True

    def test_adjust(self):
        dtype = opencl_types.dtypes[root.common.precision_type]
        inp = Vector(numpy.array([[1, 2, 3, 2, 1],
                                  [0, 1, 2, 1, 0],
                                  [0, 1, 0, 1, 0],
                                  [2, 0, 1, 0, 2],
                                  [1, 0, 1, 0, 1]], dtype=dtype))
        ra2a = ResizableAll2All(self.parent, output_sample_shape=(3,),
                                weights_stddev=0.05)
        ra2a.input = inp
        ra2a.initialize(device=self.device)
        self.assertEqual(ra2a.output.shape, (5, 3))
        ra2a.run()
        ra2a.output_sample_shape = (6,)
        ra2a.run()
        self.assertEqual(ra2a.output.shape, (5, 6))


@assign_backend("ocl")
class OpenCLTestResizableAll2All(TestResizableAll2All):
    pass


@assign_backend("cuda")
class CUDATestResizableAll2All(TestResizableAll2All):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
