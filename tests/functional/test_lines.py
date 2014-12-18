#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import unittest

from veles.config import root
import veles.backends as opencl
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.samples.Lines.lines as lines
import veles.dummy as dummy_workflow


class TestLines(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    @timeout(300)
    def test_lines(self):
        logging.info("Will test lines workflow with one convolutional relu"
                     " layer and one fully connected relu layer")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        train = os.path.join(root.common.test_dataset_root,
                             "Lines/lines_min/learn")
        valid = os.path.join(root.common.test_dataset_root,
                             "Lines/lines_min/test")
        root.common.precision_level = 1
        root.lines.update({
            "accumulator": {"bars": 30, "squash": True},
            "decision": {"fail_iterations": 100},
            "snapshotter": {"prefix": "lines_test"},
            "loader": {"minibatch_size": 60},
            "layers": [{"type": "conv_relu", "n_kernels": 32,
                        "kx": 11, "ky": 11, "sliding": (4, 4),
                        "learning_rate": 0.0003, "weights_decay": 0.0,
                        "gradient_moment": 0.9, "weights_filling": "gaussian",
                        "weights_stddev": 0.001, "bias_filling": "gaussian",
                        "bias_stddev": 0.001},

                       {"type": "max_pooling",
                        "kx": 3, "ky": 3, "sliding": (2, 2)},

                       {"type": "all2all_relu", "output_shape": 32,
                        "learning_rate": 0.0001, "weights_decay": 0.0,
                        "gradient_moment": 0.9, "weights_filling": "uniform",
                        "weights_stddev": 0.05, "bias_filling": "uniform",
                        "bias_stddev": 0.05},

                       {"type": "softmax", "output_shape": 4,
                        "learning_rate": 0.0001, "weights_decay": 0.0,
                        "gradient_moment": 0.9, "weights_filling": "uniform",
                        "weights_stddev": 0.05, "bias_filling": "uniform",
                        "bias_stddev": 0.05}],
            "path_for_load_data": {"validation": valid, "train": train}})

        self.w = lines.LinesWorkflow(dummy_workflow.DummyWorkflow(),
                                     layers=root.lines.layers,
                                     device=self.device)
        self.w.decision.max_epochs = 9
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 1
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 54)
        self.assertEqual(9, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)

        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 14
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 40)
        self.assertEqual(14, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
