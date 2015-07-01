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

Configuration file for video_ae. Model - autoencoder.

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

root.video_ae.update({
    "decision": {"fail_iterations": 100, "max_epochs": 100000},
    "downloader": {
        "url":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/VideoAE/video_ae.tar",
        "directory": root.common.dirs.datasets,
        "files": ["video_ae"]},
    "snapshotter": {"prefix": "video_ae"},
    "loader": {"minibatch_size": 50, "force_numpy": False,
               "train_paths":
               (os.path.join(root.common.dirs.datasets, "video_ae/img"),),
               "color_space": "GRAY",
               "background_color": (0x80,),
               "normalization_type": "linear"
               },
    "weights_plotter": {"limit": 16},
    "learning_rate": 0.01,
    "weights_decay": 0.00005,
    "layers": [9, [90, 160]]})
