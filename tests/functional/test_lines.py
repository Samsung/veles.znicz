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
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def init_wf(self, workflow, snapshot):
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)

        workflow.initialize(device=self.device, snapshot=snapshot)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)

    def check_write_error_rate(self, workflow, error):
        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, error)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

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

        root.lines.update({
            "loader_name": "full_batch_auto_label_file_image",
            "loss_function": "softmax",
            "decision": {"fail_iterations": 100,
                         "max_epochs": 9},
            "snapshotter": {"prefix": "lines",
                            "time_interval": 0, "interval": 9 + 1},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(
                                 root.common.cache_dir, "tmp/validation"),
                             os.path.join(
                                 root.common.cache_dir, "tmp/train")]},
            "loader": {"minibatch_size": 12, "force_cpu": False,
                       "normalization_type": "mean_disp",
                       "color_space": "RGB", "filename_types": ["jpeg"],
                       "train_paths": [train], "validation_paths": [valid]},
            "weights_plotter": {"limit": 32},
            "layers": [
                {"type": "conv_relu",
                 "->": {"n_kernels": 32, "kx": 11, "ky": 11,
                        "sliding": (4, 4), "weights_filling": "gaussian",
                        "weights_stddev": 0.001,
                        "bias_filling": "gaussian", "bias_stddev": 0.001},
                 "<-": {"learning_rate": 0.003,
                        "weights_decay": 0.0, "gradient_moment": 0.9}},
                {"type": "max_pooling",
                 "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},
                {"type": "all2all_relu",
                 "->": {"weights_filling": "uniform",
                        "bias_filling": "uniform",
                        "output_sample_shape": 32,
                        "weights_stddev": 0.05,
                        "bias_stddev": 0.05},
                 "<-": {"learning_rate": 0.001, "weights_decay": 0.0,
                        "gradient_moment": 0.9}},
                {"type": "softmax",
                 "->": {"output_sample_shape": 4, "weights_filling": "uniform",
                        "weights_stddev": 0.05, "bias_filling": "uniform",
                        "bias_stddev": 0.05},
                 "<-": {"learning_rate": 0.001, "weights_decay": 0.0,
                        "gradient_moment": 0.9}}]})

        root.common.precision_level = 1
        root.common.precision_type = "double"
        root.common.engine.backend = "ocl"

        self.w = lines.LinesWorkflow(dummy_workflow.DummyLauncher(),
                                     decision_config=root.lines.decision,
                                     snapshotter_config=root.lines.snapshotter,
                                     image_saver_config=root.lines.image_saver,
                                     loader_config=root.lines.loader,
                                     layers=root.lines.layers,
                                     loader_name=root.lines.loader_name,
                                     loss_function=root.lines.loss_function)
        # Test workflow
        self.init_wf(self.w, False)
        self.w.run()
        self.check_write_error_rate(self.w, 54)

        file_name = self.w.snapshotter.file_name

        # Test loading from snapshot
        logging.info("Will load workflow from %s" % file_name)

        self.wf = Snapshotter.import_(file_name)

        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 24
        self.wf.decision.complete <<= False

        self.init_wf(self.wf, True)
        self.wf.run()
        self.check_write_error_rate(self.wf, 46)

        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
