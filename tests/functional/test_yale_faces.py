#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os

from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.units import TrivialUnit
import veles.znicz.samples.YaleFaces.yale_faces as yale_faces
import veles.dummy as dummy_workflow
from veles.znicz.tests.functional import StandardTest


class TestYaleFaces(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.yalefaces.update({
            "decision": {"fail_iterations": 50, "max_epochs": 3},
            "loss_function": "softmax",
            "snapshotter": {"prefix": "yalefaces_test", "interval": 4,
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

    def init_and_run(self, snapshot):
        self.workflow = yale_faces.YaleFacesWorkflow(
            dummy_workflow.DummyLauncher(),
            loader_name=root.yalefaces.loader_name,
            loader_config=root.yalefaces.loader,
            decision_config=root.yalefaces.decision,
            snapshotter_config=root.yalefaces.snapshotter,
            layers=root.yalefaces.layers,
            loss_function=root.yalefaces.loss_function)
        self.init_wf(self.workflow, snapshot)
        self.workflow.run()

    errors = {"ocl": (239, 167, 233), "cuda": (222, 167, 236)}

    @timeout(1500)
    @multi_device()
    def test_yale_faces_gpu(self):
        self.info("Will test fully connectected yale_faces workflow")

        errors = self.errors[self.device.backend_name]
        self.info("Will run workflow with double")
        root.common.precision_type = "double"

        # Test workflow
        self.init_and_run(False)
        self.check_write_error_rate(self.workflow, errors[0])

        self.info("Extracting the forward workflow...")
        fwd_wf = self.workflow.extract_forward_workflow(
            loader_name=root.yalefaces.loader_name,
            loader_config=root.yalefaces.loader,
            result_unit_factory=TrivialUnit)
        self.assertEqual(len(fwd_wf.forwards), 2)

        file_name = self.workflow.snapshotter.file_name

        # Test loading from snapshot
        self.info("Will load workflow from snapshot: %s", file_name)

        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 6
        self.wf.decision.complete <<= False

        self.init_wf(self.wf, True)
        self.wf.run()
        self.check_write_error_rate(self.wf, errors[1])

        self.info("Will run workflow with float")
        root.common.precision_type = "float"

        # Test workflow with ocl and float
        self.init_and_run(False)
        self.check_write_error_rate(self.workflow, errors[2])

if __name__ == "__main__":
    StandardTest.main()
