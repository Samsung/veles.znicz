"""
Created on June 06, 2014

A base class for test cases.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
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

        assert root.common.disable_plotting
        self.seed()

    def getParent(self):
        return DummyLauncher()

    def seed(self):
        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.int32, count=1024))
        prng.get(2).seed(numpy.fromfile("%s/veles/znicz/tests/research/seed2" %
                                        root.common.veles_dir,
                                        dtype=numpy.int32, count=1024))

    def tearDown(self):
        if Unit._pool_ is not None:
            Unit._pool_.shutdown(execute_remaining=False)
            Unit._pool_ = None
        super(StandardTest, self).tearDown()

    @staticmethod
    def main():
        StandardTest.setup_logging(logging.INFO)
        unittest.main()
