# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 19, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import logging
import numpy
import os
import sys
import tarfile
import time
import unittest
from zope.interface import implementer

from veles import memory, prng
from veles.accelerated_units import IOpenCLUnit, ICUDAUnit, INumpyUnit
from veles.backends import NumpyDevice
from veles.dummy import DummyWorkflow
from veles.znicz.gd import GradientDescent
from veles.znicz.nn_units import Forward, ForwardExporter


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class TrivialForward(Forward):
    def numpy_run(self):
        pass

    def ocl_init(self):
        pass

    def ocl_run(self):
        pass

    def cuda_init(self):
        pass

    def cuda_run(self):
        pass


class Test(unittest.TestCase):
    def setUp(self):
        self.parent = DummyWorkflow()

    def tearDown(self):
        del self.parent

    def test_ocl_set_const_args(self):
        u = GradientDescent(self.parent)
        self.assertTrue(u.ocl_set_const_args)
        for attr in ("learning_rate", "weights_decay",
                     "l1_vs_l2", "gradient_moment",
                     "learning_rate_bias", "weights_decay_bias",
                     "l1_vs_l2_bias", "gradient_moment_bias"):
            vle = prng.get().rand()
            u.ocl_set_const_args = False
            setattr(u, attr, vle)
            self.assertTrue(u.ocl_set_const_args)
            self.assertEqual(getattr(u, attr), vle)

    def testForwardExporter(self):
        workflow = self.parent
        fe = ForwardExporter(workflow, prefix="testp", time_interval=0)
        fe.link_from(workflow.start_point)
        fe.suffix = "tests"
        for _ in range(3):
            fwd = TrivialForward(workflow, name="forward")
            fwd.weights.mem = numpy.ones(1000)
            fwd.bias.mem = numpy.ones(10)
            fwd.input = memory.Vector()
            fe.forwards.append(fwd)
            fwd.initialize(NumpyDevice())
        workflow.initialize(snapshot=False)
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
