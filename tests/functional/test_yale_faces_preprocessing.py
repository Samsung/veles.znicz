#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on April 1, 2015

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


import os

from veles.config import root
from veles.tests import timeout, multi_device
import veles.znicz.samples.YaleFaces.yale_faces_preprocessing as yalepreprocess
from veles.znicz.tests.functional import StandardTest


class TestYaleFacesPreprocessing(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.yalefaces.update({
            "preprocessing": True,
            "loader_name": "full_batch_auto_label_file_image",
            "datasaver": {"file_name":
                          os.path.join(root.common.dirs.datasets,
                                       "test_yale_faces_minibatches.sav")},
            "loader": {"minibatch_size": 40, "force_numpy": False,
                       "validation_ratio": 0.15,
                       "file_subtypes": ["x-portable-graymap"],
                       "ignored_files": [".*Ambient.*"],
                       "shuffle_limit": 0,
                       "add_sobel": False,
                       "normalization_type": "mean_disp",
                       "mirror": False,
                       "color_space": "GRAY",
                       "background_color": (0,),
                       "train_paths":
                       [os.path.join(root.common.dirs.datasets,
                                     "CroppedYale")]}})

    def init_wf(self, workflow, snapshot):
        workflow.initialize(device=self.device, snapshot=snapshot)

    def check_write_error_rate(self, workflow, error):
        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, error)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

    def init_and_run(self, snapshot):
        workflow = yalepreprocess.YaleFacesWorkflow(
            self.parent,
            loader_name=root.yalefaces.loader_name,
            preprocessing=root.yalefaces.preprocessing,
            data_saver_config=root.yalefaces.datasaver,
            loader_config=root.yalefaces.loader)
        self.init_wf(workflow, snapshot)
        workflow.run()
        return workflow

    @timeout(1500)
    @multi_device()
    def test_yale_faces_gpu(self):
        self.info("Will test preprocessing of yale_faces workflow")

        file_name = root.yalefaces.datasaver.file_name

        try:
            os.remove(file_name)
        except:
            pass

        root.common.precision_type = "double"

        workflow = self.init_and_run(False)
        self.assertIsNone(workflow.thread_pool.failure)

        if not os.access(file_name, os.R_OK):
            raise OSError("File %s not exist" % file_name)

if __name__ == "__main__":
    StandardTest.main()
