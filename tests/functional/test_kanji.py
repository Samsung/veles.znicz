#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import sys
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.prng as prng
from veles.tests import timeout
import veles.znicz.samples.kanji as kanji
import veles.tests.dummy_workflow as dummy_workflow


class TestKanji(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    @timeout()
    def test_kanji(self):
        logging.info("Will test kanji workflow")
        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.int32, count=1024))
        prng.get(2).seed(numpy.fromfile("%s/veles/znicz/tests/research/seed2" %
                                        root.common.veles_dir,
                                        dtype=numpy.int32, count=1024))
        root.decision.fail_iterations = -1
        root.kanji.data_paths.target = os.path.join(
            root.common.veles_dir,
            ("veles/znicz/tests/data/kanji/target/targets.%d.pickle" %
             (sys.version_info[0])))

        root.kanji.data_paths.train = os.path.join(
            root.common.veles_dir, ("veles/znicz/tests/data/kanji/train"))

        root.kanji.index_map = os.path.join(
            root.kanji.data_paths.train, "index_map.%d.pickle" %
            (sys.version_info[0]))

        w = kanji.Workflow(dummy_workflow.DummyWorkflow(),
                           layers=[30, 30, 24 * 24], device=self.device)
        w.initialize(learning_rate=root.kanji.learning_rate,
                     weights_decay=root.kanji.weights_decay,
                     minibatch_size=root.loader.minibatch_size,
                     device=self.device, weights=None, bias=None)
        w.run()
        err = w.decision.epoch_n_err[1]
        self.assertEqual(err, 18)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
