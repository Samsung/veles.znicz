#!/usr/bin/python3.3 -O
"""
Created on November 18, 2013

@author: Lyubov Podoynitsina <lyubov.p@samsung.com>
"""


import logging
import numpy
import os
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.rnd as rnd
import veles.znicz.samples.wine as wine
import veles.tests.dummy_workflow as dummy_workflow


class TestWine(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def test_wine(self):
        logging.info("Will test loader, decision, evaluator units")
        rnd.default.seed(numpy.fromfile("%s/veles/znicz/samples/seed" %
                                        (root.common.veles_dir),
                                        dtype=numpy.int32, count=1024))
        root.common.update = {"plotters_disabled": True}

        root.update = {"decision": {"fail_iterations": 200,
                                    "snapshot_prefix": "wine"},
                       "global_alpha": 0.75,
                       "global_lambda": 0.0,
                       "layers":  [8, 3],
                       "loader": {"minibatch_maxsize": 10},
                       "path_for_load_data":
                       os.path.join(root.common.veles_dir,
                                    "veles/znicz/samples/wine/wine.data")}

        w = wine.Workflow(dummy_workflow.DummyWorkflow(), layers=[8, 3],
                          device=self.device)
        w.initialize()
        w.run()
        epoch = w.decision.epoch_number[0]
        self.assertEqual(epoch, 11)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
