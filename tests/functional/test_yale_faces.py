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
import veles.znicz.samples.YaleFaces.yale_faces as yale_faces
import veles.dummy as dummy_workflow


class TestYaleFaces(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def init_wf(self, workflow, device):
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)

        workflow.initialize(device=device)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)

    def check_write_error_rate(self, workflow, error):
        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, error)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

    def init_and_run(self, device):
        self.w = yale_faces.YaleFacesWorkflow(
            dummy_workflow.DummyLauncher(),
            loader_name=root.yalefaces.loader_name,
            loader_config=root.yalefaces.loader,
            decision_config=root.yalefaces.decision,
            snapshotter_config=root.yalefaces.snapshotter,
            layers=root.yalefaces.layers,
            loss_function=root.yalefaces.loss_function,
            device=device)
        self.init_wf(self.w, device)
        self.w.run()

    @timeout(1500)
    def test_yale_faces_gpu(self):
        logging.info("Will test fully connectected yale_faces workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.common.precision_level = 1

        root.yalefaces.update({
            "decision": {"fail_iterations": 50, "max_epochs": 3},
            "loss_function": "softmax",
            "snapshotter": {"prefix": "yalefaces_test", "interval": 3,
                            "time_interval": 0},
            "loader_name": "full_batch_auto_label_file_image",
            "loader": {"minibatch_size": 40, "force_cpu": False,
                       "validation_ratio": 0.15,
                       "filename_types": ["x-portable-graymap"],
                       "ignored_files": [".*Ambient.*"],
                       "shuffle_limit": numpy.iinfo(numpy.uint32).max,
                       "add_sobel": False,
                       "mirror": False,
                       "color_space": "GRAY",
                       "background_color": (0,),
                       "normalization_type": "mean_disp",
                       "train_paths":
                           [os.path.join(root.common.test_dataset_root,
                                         "CroppedYale")]},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": 100},
                        "<-": {"learning_rate": 0.01,
                               "weights_decay": 0.00005}},
                       {"type": "softmax",
                        "<-": {"learning_rate": 0.01,
                               "weights_decay": 0.00005}}]})

        self._test_yale_faces_gpu(device=self.device)
        self._test_mnist_ae_cpu(None)
        logging.info("All Ok")

    def _test_yale_faces_gpu(self, device):
        logging.info("Will run workflow with double and ocl backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        # Test workflow
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 239)

        file_name = self.w.snapshotter.file_name

        # Test loading from snapshot
        logging.info("Will load workflow from snapshot: %s" % file_name)

        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 6
        self.wf.decision.complete <<= False

        self.init_wf(self.wf, device)
        self.wf.run()
        self.check_write_error_rate(self.wf, 167)

        logging.info("Will run workflow with double and cuda backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "cuda"}})

        # Test workflow with cuda and double
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 222)

        logging.info("Will run workflow with float and ocl backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "float",
            "engine": {"backend": "ocl"}})

        # Test workflow with ocl and float
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 233)

        logging.info("Will run workflow with float and cuda backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "float",
            "engine": {"backend": "cuda"}})

        # Test workflow with cuda and float
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 236)

    def _test_mnist_ae_cpu(self, device):
        logging.info("Will run workflow with --disable-acceleration")

        # Test workflow with --disable-acceleration
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 227)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
