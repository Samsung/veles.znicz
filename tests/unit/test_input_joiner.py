"""
Created on Oct 29, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import unittest
import input_joiner
import opencl
import formats
import numpy


class TestInputJoiner(unittest.TestCase):
    def do_test(self, device):
        a = formats.Vector()
        a.v = numpy.arange(25, dtype=numpy.float32)
        b = formats.Vector()
        b.v = numpy.arange(5, dtype=numpy.float32)
        c = formats.Vector()
        c.v = numpy.arange(35, dtype=numpy.float32)
        a.initialize(device)
        b.initialize(device)
        c.initialize(device)
        obj = input_joiner.InputJoiner(inputs=[a, b, c], device=device)
        obj.initialize()
        obj.run()
        obj.output.sync()
        for i in range(len(a.v)):
            self.assertEqual(a.v[i], obj.output.v[i], "Failed")
        for i in range(len(b.v)):
            self.assertEqual(b.v[i],
                obj.output.v[a.v.size + i], "Failed")
        for i in range(len(c.v)):
            self.assertEqual(c.v[i],
                obj.output.v[a.v.size + b.v.size + i], "Failed")

    def do_test2(self, device):
        a = formats.Vector()
        a.v = numpy.arange(25, dtype=numpy.float32)
        b = formats.Vector()
        b.v = numpy.arange(5, dtype=numpy.float32)
        c = formats.Vector()
        c.v = numpy.arange(35, dtype=numpy.float32)
        a.initialize(device)
        b.initialize(device)
        c.initialize(device)
        obj = input_joiner.InputJoiner(inputs=[a, b, c], output_shape=[80],
                                       device=device)
        obj.initialize()
        obj.run()
        obj.output.sync()
        for i in range(len(a.v)):
            self.assertEqual(a.v[i], obj.output.v[i], "Failed")
        for i in range(len(b.v)):
            self.assertEqual(b.v[i],
                obj.output.v[a.v.size + i], "Failed")
        for i in range(len(c.v)):
            self.assertEqual(c.v[i],
                obj.output.v[a.v.size + b.v.size + i], "Failed")
        for i in range(len(obj.output.v) - a.v.size - b.v.size - c.v.size):
            self.assertEqual(0,
                obj.output.v[a.v.size + b.v.size + c.v.size + i], "Failed")

    def do_test3(self, device):
        a = formats.Vector()
        a.v = numpy.arange(25, dtype=numpy.float32)
        b = formats.Vector()
        b.v = numpy.arange(5, dtype=numpy.float32)
        c = formats.Vector()
        c.v = numpy.arange(35, dtype=numpy.float32)
        a.initialize(device)
        b.initialize(device)
        c.initialize(device)
        obj = input_joiner.InputJoiner(inputs=[a, b, c], output_shape=[50],
                                       device=device)
        obj.initialize()
        obj.run()
        obj.output.sync()
        for i in range(len(a.v)):
            self.assertEqual(a.v[i], obj.output.v[i], "Failed")
        for i in range(len(b.v)):
            self.assertEqual(b.v[i],
                obj.output.v[a.v.size + i], "Failed")
        for i in range(len(c.v)):
            if a.v.size + b.v.size + i >= len(obj.output.v):
                break
            self.assertEqual(c.v[i],
                obj.output.v[a.v.size + b.v.size + i], "Failed")

    def testGPU(self):
        print("Will test InputJoiner() on GPU.")
        cl = opencl.DeviceList()
        device = cl.get_device()
        self.do_test(device)

    def testCPU(self):
        print("Will test InputJoiner() on CPU.")
        self.do_test(None)

    def testGPU2(self):
        print("Will test InputJoiner() on GPU "
              "with output size greater than inputs.")
        cl = opencl.DeviceList()
        device = cl.get_device()
        self.do_test2(device)

    def testCPU2(self):
        print("Will test InputJoiner() on CPU "
              "with output size greater than inputs.")
        self.do_test2(None)

    def testGPU3(self):
        print("Will test InputJoiner() on GPU "
              "with output size less than inputs.")
        cl = opencl.DeviceList()
        device = cl.get_device()
        self.do_test3(device)

    def testCPU3(self):
        print("Will test InputJoiner() on CPU "
              "with output size less than inputs.")
        self.do_test3(None)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
