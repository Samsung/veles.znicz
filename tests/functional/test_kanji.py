#!/usr/bin/python3.3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import six
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.rnd as rnd
import veles.znicz.samples.kanji as kanji
import veles.tests.dummy_workflow as dummy_workflow


class TestKanji(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def test_kanji(self):
        logging.info("Will test loader, decision, evaluator units")
        rnd.default.seed(numpy.fromfile("%s/veles/znicz/samples/seed"
                                        % (root.common.veles_dir),
                                        dtype=numpy.int32, count=1024))
        rnd.default2.seed(numpy.fromfile("%s/veles/znicz/samples/seed2"
                                         % (root.common.veles_dir),
                                         dtype=numpy.int32, count=1024))
        root.decision.fail_iterations = -1
        root.kanji.path_for_load_data.target = os.path.join(
            root.common.veles_dir,
            ("veles/znicz/tests/data/kanji/target/targets.%d.pickle" %
             (3 if six.PY3 else 2)))

        root.kanji.path_for_load_data.train = os.path.join(
            root.common.veles_dir, ("veles/znicz/tests/data/kanji/train"))

        root.kanji.index_map = os.path.join(
            root.kanji.path_for_load_data.train, "index_map.%d.pickle" %
            (3 if six.PY3 else 2))

        w = kanji.Workflow(dummy_workflow.DummyWorkflow(),
                           layers=[30, 30, 24 * 24], device=self.device)
        w.initialize(global_alpha=root.kanji.global_alpha,
                     global_lambda=root.kanji.global_lambda,
                     minibatch_maxsize=root.loader.minibatch_maxsize,
                     device=self.device, weights=None, bias=None)
        w.run()
        err = w.decision.epoch_n_err[1]
        self.assertEqual(err, 17)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
