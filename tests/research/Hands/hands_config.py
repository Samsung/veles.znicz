# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mart 26, 2014

Configuration file for hands.
Model - fully-connected Neural Network with SoftMax loss function.

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


train_dir = [os.path.join(root.common.datasets_root, "hands/Training")]
validation_dir = [os.path.join(root.common.datasets_root, "hands/Testing")]


root.hands.update({
    "decision": {"fail_iterations": 100, "max_epochs": 10000},
    "loss_function": "softmax",
    "downloader": {
        "url":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/Hands/hands.tar",
        "directory": root.common.datasets_root,
        "files": ["hands"]},
    "image_saver": {"do": True,
                    "out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader_name": "hands_loader",
    "snapshotter": {"prefix": "hands", "interval": 1, "time_interval": 0},
    "loader": {"minibatch_size": 40, "train_paths": train_dir,
               "force_numpy": False, "color_space": "GRAY",
               "background_color": (0,),
               "normalization_type": "linear",
               "validation_paths": validation_dir},
    "layers": [{"type": "all2all_tanh",
                "->": {"output_sample_shape": 30},
                "<-": {"learning_rate": 0.008, "weights_decay": 0.0}},
               {"type": "softmax",
                "<-": {"learning_rate": 0.008, "weights_decay": 0.0}}]})
