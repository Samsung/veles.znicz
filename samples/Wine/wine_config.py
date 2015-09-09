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


root.wine.update({
    "decision": {"fail_iterations": 200,
                 "max_epochs": 100},
    "downloader": {
        "url":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/WineRelu/wine.tar",
        "directory": root.common.dirs.datasets,
        "files": ["wine"]},
    "snapshotter": {"prefix": "wine", "interval": 10, "time_interval": 0},
    "loader": {"minibatch_size": 10,
               "dataset_file":
               os.path.join(root.common.dirs.datasets, "wine/wine.txt.gz"),
               "force_numpy": False},
    "learning_rate": 0.3,
    "weights_decay": 0.0,
    "layers": [8, 3]})
