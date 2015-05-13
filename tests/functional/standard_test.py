# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 06, 2014

A base class for test cases.

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


# pylint: disable=W0633

import logging
import numpy
import os
import unittest

from veles.config import root
from veles.dummy import DummyLauncher
from veles.tests import AcceleratedTest
import veles.prng as prng
from veles.units import Unit


class StandardTest(AcceleratedTest):
    def setUp(self):
        super(StandardTest, self).setUp()
        self.data_dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data")
        root.common.update({
            "precision_level": 1,
            "precision_type": "double"})

        assert root.common.disable.plotting
        self.seed()

    def getParent(self):
        return DummyLauncher()

    def seed(self):
        prng.get().seed(numpy.fromfile("%s/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.int32, count=1024))
        prng.get(2).seed(numpy.fromfile("%s/znicz/tests/research/seed2" %
                                        root.common.veles_dir,
                                        dtype=numpy.int32, count=1024))

    def tearDown(self):
        if Unit._pool_ is not None:
            Unit._pool_.shutdown(execute_remaining=False, force=True)
            Unit._pool_ = None
        super(StandardTest, self).tearDown()

    @staticmethod
    def main():
        StandardTest.setup_logging(logging.INFO)
        logging.getLogger("ThreadPool").setLevel(logging.DEBUG)
        unittest.main()
