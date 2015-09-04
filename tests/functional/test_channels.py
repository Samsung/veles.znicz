#!/usr/bin/env python3
# -*-coding: utf-8 -*-
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
import numpy
import os
from six import PY3

from veles.config import root
from veles.memory import Array
from veles.snapshotter import SnapshotterToFile
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.TvChannels.channels as channels


class TestChannels(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.channels.update({
            "decision": {"fail_iterations": 50,
                         "max_epochs": 3},
            "downloader": {
                "url":
                "https://s3-eu-west-1.amazonaws.com/veles.forge/"
                "TvChannels/train.tar",
                "directory": root.common.dirs.datasets,
                "files": ["train"]},
            "snapshotter": {"prefix": "test_channels", "interval": 4,
                            "time_interval": 0},
            "image_saver": {"out_dirs": [
                os.path.join(root.common.dirs.cache, "tmp/test"),
                os.path.join(root.common.dirs.cache,
                             "tmp/validation"),
                os.path.join(root.common.dirs.cache,
                             "tmp/train")]},
            "loader": {"minibatch_size": 30,
                       "force_numpy": True,
                       "validation_ratio": 0.15,
                       "shuffle_limit": numpy.iinfo(numpy.uint32).max,
                       "normalization_type": "mean_disp",
                       "add_sobel": True,
                       "file_subtypes": ["png"],
                       "background_image":
                       numpy.zeros([64, 64, 4], dtype=numpy.uint8),
                       "mirror": False,
                       "color_space": "HSV",
                       "background_color": (0, 0, 0, 0),
                       "scale": (64, 64),
                       "scale_maintain_aspect_ratio": True,
                       "train_paths":
                       [os.path.join(root.common.dirs.datasets, "train")]},
            "loss_function": "softmax",
            "loader_name": "full_batch_auto_label_file_image",
            "layers": [{"type": "all2all_tanh",
                        "<-": {"learning_rate": 0.01,
                               "weights_decay": 0.00005,
                               "weights_decay_bias": 0.0005,
                               "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9,
                               "learning_rate_bias": 0.02},
                        "->": {"output_sample_shape": 100}},
                       {"type": "softmax",
                        "->": {"output_sample_shape": 8},
                        "<-": {"learning_rate": 0.01,
                               "weights_decay_bias": 0.0005,
                               "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9,
                               "learning_rate_bias": 0.02,
                               "weights_decay": 0.00005}}]})

    @timeout(800)
    @multi_device()
    def test_channels_all2all(self):
        self.info("Will test channels fully connected workflow")

        workflow = channels.ChannelsWorkflow(
            self.parent,
            decision_config=root.channels.decision,
            snapshotter_config=root.channels.snapshotter,
            image_saver_config=root.channels.image_saver,
            loader_config=root.channels.loader,
            downloader_config=root.channels.downloader,
            layers=root.channels.layers,
            loader_name=root.channels.loader_name,
            loss_function=root.channels.loss_function)

        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(
            device=self.device,
            minibatch_size=root.channels.loader.minibatch_size,
            snapshot=False)

        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        file_name = workflow.snapshotter.destination

        err = workflow.decision.epoch_n_err[1]
        # Varies depending on Pillow version (assumed)
        self.assertLessEqual(err, 12)
        self.assertGreaterEqual(err, 11)
        self.assertEqual(3, workflow.loader.epoch_number)

        # Garbage collection
        del workflow
        self.parent = self.getParent()
        if PY3:
            Array.reset_all()
        gc.collect()

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = SnapshotterToFile.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 4
        workflow_from_snapshot.decision.complete <<= False
        self.assertTrue(workflow_from_snapshot.loader.force_numpy)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(
            device=self.device,
            minibatch_size=root.channels.loader.minibatch_size,
            snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()
        self.assertIsNone(workflow_from_snapshot.thread_pool.failure)

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        # PIL Image for python2 and PIL for python3 can return different values
        self.assertEqual(err, 10)
        self.assertEqual(4, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")


if __name__ == "__main__":
    StandardTest.main()
