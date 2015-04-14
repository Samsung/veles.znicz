# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 13, 2014

Configuration file for yale-faces (Self-constructing Model).
Model - fully connected neural network.

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


root.yalefaces.update({
    "downloader": {
        "url":
        "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip",
        "directory": root.common.test_dataset_root,
        "files": ["CroppedYale"]},
    "decision": {"fail_iterations": 50, "max_epochs": 1000},
    "loss_function": "softmax",
    "loader_name": "full_batch_auto_label_file_image",
    "snapshotter": {"prefix": "yalefaces", "interval": 10, "time_interval": 0},
    "loader": {"minibatch_size": 40, "force_numpy": False,
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
               [os.path.join(root.common.test_dataset_root, "CroppedYale")]},
    "layers": [{"type": "all2all_tanh",
                "->": {"output_sample_shape": 100},
                "<-": {"learning_rate": 0.01, "weights_decay": 0.00005}},
               {"type": "softmax",
                "<-": {"learning_rate": 0.01, "weights_decay": 0.00005}}]})
