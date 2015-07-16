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
import unittest
from zope.interface import implementer

from veles.accelerated_units import IOpenCLUnit, ICUDAUnit, INumpyUnit
from veles.dummy import DummyWorkflow
from veles.znicz.nn_units import Forward, NNSnapshotterToFile


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

    def test_nnsnapshotter(self):
        nns = NNSnapshotterToFile(self.parent)
        nns.suffix = "suffix"
        nns.initialize()
        nns.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
