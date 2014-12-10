"""
Created on May 19, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import logging
import numpy
import os
import sys
import tarfile
import time
import unittest
from zope.interface import implementer

from veles.dummy import DummyWorkflow
from veles.znicz.nn_units import Forward, ForwardExporter
from veles.znicz.gd import GradientDescent
from veles.accelerated_units import IOpenCLUnit
from veles import memory


@implementer(IOpenCLUnit)
class TrivialForward(Forward):
    def cpu_run(self):
        pass

    def ocl_run(self):
        pass


class Test(unittest.TestCase):
    def test_ocl_set_const_args(self):
        u = GradientDescent(DummyWorkflow())
        self.assertTrue(u.ocl_set_const_args)
        for attr in ("learning_rate", "weights_decay",
                     "l1_vs_l2", "gradient_moment",
                     "learning_rate_bias", "weights_decay_bias",
                     "l1_vs_l2_bias", "gradient_moment_bias"):
            vle = numpy.random.rand()
            u.ocl_set_const_args = False
            setattr(u, attr, vle)
            self.assertTrue(u.ocl_set_const_args)
            self.assertEqual(getattr(u, attr), vle)

    def testForwardExporter(self):
        workflow = DummyWorkflow()
        fe = ForwardExporter(workflow, prefix="testp", time_interval=0)
        fe.suffix = "tests"
        for _ in range(3):
            fwd = TrivialForward(workflow, name="forward")
            fwd.weights.mem = numpy.ones(1000)
            fwd.bias.mem = numpy.ones(10)
            fwd.input = memory.Vector()
            fe.forwards.append(fwd)
            fwd.initialize(None)
        workflow.initialize()
        fe.run()
        self.assertTrue(fe.file_name)
        self.assertTrue(os.path.exists(fe.file_name))
        self.assertTrue(os.path.exists(os.path.join(
            os.path.dirname(fe.file_name), "testp_current_wb.%d.tar.gz" %
            (sys.version_info[0]))))
        self.assertLess(fe.time, time.time())
        self.assertEqual("testp_tests_wb.%d.tar.gz" % (sys.version_info[0]),
                         os.path.basename(fe.file_name))
        with tarfile.open(fe.file_name, "r:gz") as tar:
            for index in ("001", "002", "003"):
                try:
                    # On Python 2.7, TarFile does not have __exit__()
                    fileobj = tar.extractfile("%s_forward.npz" % index)
                    with numpy.load(fileobj) as npz:
                        weights, bias = npz["weights"], npz["bias"]
                    self.assertTrue(all(weights == 1))
                    self.assertTrue(all(bias == 1))
                finally:
                    fileobj.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
