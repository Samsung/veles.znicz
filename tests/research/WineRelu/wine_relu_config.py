# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mart 21, 2014

Configuration file for Wine.
Model - fully-connected Neural Network with SoftMax loss function with RELU
activation.

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


root.wine_relu.update({
    "decision": {"fail_iterations": 250, "max_epochs": 100000},
    "downloader": {
        "url":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/WineRelu/wine.tar",
        "directory": root.common.dirs.datasets,
        "files": ["wine"]},
    "snapshotter": {"prefix": "wine_relu", "interval": 1, "time_interval": 0},
    "loader_name": "wine_loader",
    "loader": {"minibatch_size": 10, "force_numpy": False,
               "dataset_file":
               os.path.join(root.common.dirs.datasets, "wine/wine.txt.gz")},
    "layers": [{"type": "all2all_relu",
                "->": {"output_sample_shape": 10},
                "<-": {"learning_rate": 0.03, "weights_decay": 0.0}},
               {"type": "softmax",
                "<-": {"learning_rate": 0.03, "weights_decay": 0.0}}]})
