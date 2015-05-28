# -*- coding: utf-8 -*-
# !/usr/bin/python3 -O
"""
.. invisible:
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

import gc
import os
from six import PY3

from veles.backends import CUDADevice
from veles.config import root
from veles.memory import Array
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
import veles.znicz.samples.Lines.lines as lines
from veles.znicz.tests.functional import StandardTest


class TestLines(StandardTest):
    @classmethod
    def setUpClass(cls):
        train = os.path.join(root.common.datasets_root,
                             "Lines/lines_min/learn")

        valid = os.path.join(root.common.datasets_root,
                             "Lines/lines_min/test")

        root.lines.mcdnnic_parameters = {
            "<-": {"learning_rate": 0.03}}

        root.lines.update({
            "loader_name": "full_batch_auto_label_file_image",
            "loss_function": "softmax",
            "downloader": {
                "url":
                "https://s3-eu-west-1.amazonaws.com/veles.forge/"
                "Lines/lines_min.tar",
                "directory": root.common.datasets_root,
                "files": ["lines_min"]},
            "mcdnnic_topology": "12x256x256-32C4-MP2-64C4-MP3-32N-4N",
            "decision": {"fail_iterations": 100,
                         "max_epochs": 3},
            "snapshotter": {"prefix": "lines",
                            "interval": 3, "time_interval": 0},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/validation"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/train")]},
            "loader": {"minibatch_size": 12, "force_numpy": True,
                       "color_space": "RGB", "file_subtypes": ["jpeg"],
                       "normalization_type": "mean_disp",
                       "train_paths": [train],
                       "validation_paths": [valid]},
            "weights_plotter": {"limit": 32}})

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

    @timeout(1200)
    @multi_device()
    def test_lines(self):
        self.info("Will test lines workflow with one convolutional relu"
                  " layer and one fully connected relu layer")

        workflow = lines.LinesWorkflow(
            self.parent,
            decision_config=root.lines.decision,
            snapshotter_config=root.lines.snapshotter,
            image_saver_config=root.lines.image_saver,
            loader_config=root.lines.loader,
            loader_name=root.lines.loader_name,
            downloader_config=root.lines.downloader,
            loss_function=root.lines.loss_function,
            mcdnnic_topology=root.lines.mcdnnic_topology,
            mcdnnic_parameters=root.lines.mcdnnic_parameters)

        # Test workflow
        self.init_wf(workflow, False)
        workflow.run()
        self.check_write_error_rate(workflow, 47)

        file_name = workflow.snapshotter.file_name
        del workflow
        if PY3:
            Array.reset_all()
        self.parent = self.getParent()
        gc.collect()

        # Test loading from snapshot
        self.info("Will load workflow from %s", file_name)

        workflow_from_snapshot = Snapshotter.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 4
        workflow_from_snapshot.decision.complete <<= False

        self.init_wf(workflow_from_snapshot, True)
        workflow_from_snapshot.run()
        self.check_write_error_rate(workflow_from_snapshot, 41)

        self.info("All Ok")
        if not PY3 and isinstance(self.device, CUDADevice):
            # Python 2 does not free the memory properly, so we get
            # CL_OUT_OF_HOST_MEMORY etc. on the second run
            return True


if __name__ == "__main__":
    StandardTest.main()
