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

Configuration file for approximator.

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


# optional parameters

target_dir = [os.path.join(root.common.datasets_root,
                           "approximator/all_org_apertures.mat")]
train_dir = [os.path.join(root.common.datasets_root,
                          "approximator/all_dec_apertures.mat")]

root.approximator.update({
    "decision": {"fail_iterations": 1000, "max_epochs": 1000000000},
    "snapshotter": {"prefix": "approximator"},
    "loader": {"minibatch_size": 100, "train_paths": train_dir,
               "target_paths": target_dir,
               "normalization_type": "mean_disp",
               "target_normalization_type": "mean_disp"},
    "learning_rate": 0.0001,
    "weights_decay": 0.00005,
    "layers": [810, 9]})
