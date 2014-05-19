"""
Created on May 19, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import logging
import numpy
import six
import os
import tarfile
import time
import unittest

from veles.tests import DummyWorkflow
from veles.znicz.nn_units import Forward, ForwardExporter


class Test(unittest.TestCase):
    def testForwardExporter(self):
        workflow = DummyWorkflow()
        fe = ForwardExporter(workflow, prefix="testp", time_interval=0)
        fe.suffix = "tests"
        for _ in range(3):
            fwd = Forward(workflow, name="forward")
            fwd.weights.v = numpy.ones(1000)
            fwd.bias.v = numpy.ones(10)
            fe.forwards.append(fwd)
        fe.initialize()
        fe.run()
        self.assertTrue(fe.file_name)
        self.assertTrue(os.path.exists(fe.file_name))
        self.assertTrue(os.path.exists(os.path.join(
            os.path.dirname(fe.file_name), "testp_current_wb.%d.tar.gz" %
            (3 if six.PY3 else 2))))
        self.assertLess(fe.time, time.time())
        self.assertEqual("testp_tests_wb.%d.tar.gz" % (3 if six.PY3 else 2),
                         os.path.basename(fe.file_name))
        with tarfile.open(fe.file_name, "r:gz") as tar:
            for index in ("001", "002", "003"):
                with tar.extractfile("%s_forward.npz" % index) as fileobj:
                    with numpy.load(fileobj) as npz:
                        weights, bias = npz["weights"], npz["bias"]
                    self.assertTrue(all(weights == 1))
                    self.assertTrue(all(bias == 1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()