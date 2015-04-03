#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on April 2, 2014

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


import numpy
import os

from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.units import TrivialUnit
import veles.znicz.samples.YaleFaces.yale_faces as yale_faces
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
                       "file_subtypes": ["x-portable-graymap"],
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
        workflow = yale_faces.YaleFacesWorkflow(
            self.parent,
            loader_name=root.yalefaces.loader_name,
            loader_config=root.yalefaces.loader,
            decision_config=root.yalefaces.decision,
            snapshotter_config=root.yalefaces.snapshotter,
            layers=root.yalefaces.layers,
            loss_function=root.yalefaces.loss_function)
        self.init_wf(workflow, snapshot)
        workflow.run()
        return workflow

    @timeout(1500)
    @multi_device()
    def test_yale_faces_gpu(self):
        self.info("Will test fully connectected yale_faces workflow")

        errors = (239, 167, 222)
        self.info("Will run workflow with double")
        root.common.precision_type = "double"

        # Test workflow
        workflow = self.init_and_run(False)
        self.assertIsNone(workflow.thread_pool.failure)
        self.check_write_error_rate(workflow, errors[0])

        self.info("Extracting the forward workflow...")
        fwd_wf = workflow.extract_forward_workflow(
            loader_name=root.yalefaces.loader_name,
            loader_config=root.yalefaces.loader,
            result_unit_factory=TrivialUnit)
        self.assertEqual(len(fwd_wf.forwards), 2)

        file_name = workflow.snapshotter.file_name

        # Test loading from snapshot
        self.info("Will load workflow from snapshot: %s", file_name)

        workflow_from_snapshot = Snapshotter.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 6
        workflow_from_snapshot.decision.complete <<= False

        self.init_wf(workflow_from_snapshot, True)
        workflow_from_snapshot.run()
        self.check_write_error_rate(workflow_from_snapshot, errors[1])

        self.info("Will run workflow with float")
        root.common.precision_type = "float"

        # Test workflow with ocl and float
        workflow = self.init_and_run(False)
        self.assertIsNone(workflow.thread_pool.failure)
        self.check_write_error_rate(workflow, errors[2])

if __name__ == "__main__":
    StandardTest.main()
