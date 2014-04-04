#!/usr/bin/python3.3 -O
"""
Created on November 18, 2013

@author: Lyubov Podoynitsina <lyubov.p@samsung.com>
"""


import logging
import numpy
import os
import six
import unittest

from veles.config import root
import veles.launcher as launcher
import veles.opencl as opencl
import veles.rnd as rnd
import veles.znicz.samples.kanji as kanji


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
        root.common.update = {"plotters_disabled": True}
        root.decision.fail_iterations = -1
        root.path_for_target_data = os.path.join(
            root.common.veles_dir,
            ("veles/znicz/tests/data/kanji/target/targets.%d.pickle" %
             (3 if six.PY3 else 2)))

        root.path_for_train_data = os.path.join(
            root.common.veles_dir, ("veles/znicz/tests/data/kanji/train"))

        root.index_map = os.path.join(root.path_for_train_data,
                                      "index_map.%d.pickle" % (3 if
                                      six.PY3 else 2))

        l = launcher.Launcher()
        w = kanji.Workflow(l, layers=[30, 30, 24 * 24], device=self.device)
        w.initialize(global_alpha=root.global_alpha,
                     global_lambda=root.global_lambda,
                     minibatch_maxsize=root.loader.minibatch_maxsize,
                     device=self.device, weights=None, bias=None)
        l.run()
        err = w.decision.epoch_n_err[1]
        self.assertEqual(err, 17)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
