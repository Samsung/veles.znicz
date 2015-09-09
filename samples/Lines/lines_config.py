# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 6, 2014

Configuration file for lines.
Model - convolutional neural network.

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


train = os.path.join(root.common.dirs.datasets, "lines_min/learn")
valid = os.path.join(root.common.dirs.datasets, "lines_min/test")

root.lines.mcdnnic_parameters = {"<-": {"learning_rate": 0.01}}

root.lines.update({
    "loader_name": "full_batch_auto_label_file_image",
    "loss_function": "softmax",
    "downloader": {
        "url":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/Lines/lines_min.tar",
        "directory": root.common.dirs.datasets,
        "files": ["lines_min"]},
    "mcdnnic_topology": "12x256x256-32C4-MP2-64C4-MP3-32N-4N",
    "decision": {"fail_iterations": 100,
                 "max_epochs": numpy.iinfo(numpy.uint32).max},
    "snapshotter": {"prefix": "lines", "interval": 1, "time_interval": 0},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.dirs.cache, "tmp/test"),
                     os.path.join(root.common.dirs.cache, "tmp/validation"),
                     os.path.join(root.common.dirs.cache, "tmp/train")]},
    "loader": {"minibatch_size": 12, "force_numpy": False,
               "color_space": "RGB", "file_subtypes": ["jpeg"],
               "normalization_type": "mean_disp",
               "train_paths": [train],
               "validation_paths": [valid]},
    "weights_plotter": {"limit": 32, "split_channels": False}})
