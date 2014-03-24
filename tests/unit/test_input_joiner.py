"""
Created on Oct 29, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.znicz.input_joiner as input_joiner
from veles.znicz.tests.unit.dummy_workflow import DummyWorkflow


class TestInputJoiner(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def _do_tst(self, device):
        a = formats.Vector()
        a.v = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = formats.Vector()
        b.v = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = formats.Vector()
        c.v = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        a.initialize(device)
        b.initialize(device)
        c.initialize(device)
        obj = input_joiner.InputJoiner(DummyWorkflow(), inputs=[a, b, c],
                                       device=device)
        obj.initialize()
        obj.run()
        obj.output.map_read()
        nz = numpy.count_nonzero(numpy.equal(a.v,
            obj.output.v[:, :a.v.shape[1]]))
        self.assertEqual(nz, a.v.size, "Failed")
        nz = numpy.count_nonzero(numpy.equal(b.v,
            obj.output.v[:, a.v.shape[1]:a.v.shape[1] + b.v.shape[1]]))
        self.assertEqual(nz, b.v.size, "Failed")
        nz = numpy.count_nonzero(numpy.equal(c.v,
            obj.output.v[:, a.v.shape[1] + b.v.shape[1]:]))
        self.assertEqual(nz, c.v.size, "Failed")

    def _do_tst2(self, device):
        a = formats.Vector()
        a.v = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = formats.Vector()
        b.v = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = formats.Vector()
        c.v = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        a.initialize(device)
        b.initialize(device)
        c.initialize(device)
        obj = input_joiner.InputJoiner(DummyWorkflow(), inputs=[a, b, c],
                                       output_sample_shape=[80],
                                       device=device)
        obj.initialize()
        obj.run()
        obj.output.map_read()
        nz = numpy.count_nonzero(numpy.equal(a.v,
            obj.output.v[:, :a.v.shape[1]]))
        self.assertEqual(nz, a.v.size, "Failed")
        nz = numpy.count_nonzero(numpy.equal(b.v,
            obj.output.v[:, a.v.shape[1]:a.v.shape[1] + b.v.shape[1]]))
        self.assertEqual(nz, b.v.size, "Failed")
        nz = numpy.count_nonzero(numpy.equal(c.v,
            obj.output.v[:, a.v.shape[1] + b.v.shape[1]:
                         a.v.shape[1] + b.v.shape[1] + c.v.shape[1]]))
        self.assertEqual(nz, c.v.size, "Failed")
        nz = numpy.count_nonzero(
            obj.output.v[:, a.v.shape[1] + b.v.shape[1] + c.v.shape[1]:])
        self.assertEqual(nz, 0, "Failed")

    def _do_tst3(self, device):
        a = formats.Vector()
        a.v = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = formats.Vector()
        b.v = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = formats.Vector()
        c.v = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        a.initialize(device)
        b.initialize(device)
        c.initialize(device)
        obj = input_joiner.InputJoiner(DummyWorkflow(), inputs=[a, b, c],
                                       output_sample_shape=[50],
                                       device=device)
        obj.initialize()
        obj.run()
        obj.output.map_read()
        nz = numpy.count_nonzero(numpy.equal(a.v,
            obj.output.v[:, :a.v.shape[1]]))
        self.assertEqual(nz, a.v.size, "Failed")
        nz = numpy.count_nonzero(numpy.equal(b.v,
            obj.output.v[:, a.v.shape[1]:a.v.shape[1] + b.v.shape[1]]))
        self.assertEqual(nz, b.v.size, "Failed")
        nz = numpy.count_nonzero(numpy.equal(c.v[:, :obj.output.v.shape[1] -
            (a.v.shape[1] + b.v.shape[1])],
            obj.output.v[:, a.v.shape[1] + b.v.shape[1]:]))
        self.assertEqual(nz, obj.output.v.shape[0] * (obj.output.v.shape[1] -
            (a.v.shape[1] + b.v.shape[1])), "Failed")

    def testGPU(self):
        print("Will test InputJoiner() on GPU.")
        self._do_tst(self.device)

    def testCPU(self):
        print("Will test InputJoiner() on CPU.")
        self._do_tst(None)

    def testGPU2(self):
        print("Will test InputJoiner() on GPU "
              "with output size greater than inputs.")
        self._do_tst2(self.device)

    def testCPU2(self):
        print("Will test InputJoiner() on CPU "
              "with output size greater than inputs.")
        self._do_tst2(None)

    def testGPU3(self):
        print("Will test InputJoiner() on GPU "
              "with output size less than inputs.")
        self._do_tst3(self.device)

    def testCPU3(self):
        print("Will test InputJoiner() on CPU "
              "with output size less than inputs.")
        self._do_tst3(None)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
