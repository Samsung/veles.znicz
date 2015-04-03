#!/usr/bin/python3 -O
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Mart 21, 2014

Configuration file for Mnist. Model - fully-connected
Neural Network with MSE loss function with target encoded as ideal image
(784 points).

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


root.mnist784.update({
    "decision": {"fail_iterations": 100, "max_epochs": 100000},
    "snapshotter": {"prefix": "mnist_784"},
    "loader": {"minibatch_size": 100, "force_cpu": False,
               "normalization_type": "linear",
               "target_normalization_type": "linear"},
    "weights_plotter": {"limit": 16},
    "learning_rate": 0.00001,
    "weights_decay": 0.00005,
    "layers": [784, 784],
    "data_paths": {"arial": os.path.join(root.common.test_dataset_root,
                                         "arial.ttf")}})
