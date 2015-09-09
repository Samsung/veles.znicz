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

Configuration file for kanji.
Model – fully-connected Neural Network with MSE loss function.

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


train_path = os.path.join(root.common.dirs.datasets, "kanji/train")
target_path = os.path.join(root.common.dirs.datasets, "kanji/target")


root.kanji.update({
    "decision": {"fail_iterations": 1000,
                 "max_epochs": 10000},
    "downloader": {
        "url":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/Kanji/kanji.tar",
        "directory": root.common.dirs.datasets,
        "files": ["kanji"]},
    "loss_function": "mse",
    "loader_name": "full_batch_auto_label_file_image_mse",
    "add_plotters": True,
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.dirs.cache, "tmp/test"),
                     os.path.join(root.common.dirs.cache, "tmp/validation"),
                     os.path.join(root.common.dirs.cache, "tmp/train")]},
    "loader": {"minibatch_size": 50,
               "force_numpy": False,
               "file_subtypes": ["png"],
               "train_paths": [train_path],
               "target_paths": [target_path],
               "color_space": "GRAY",
               "normalization_type": "linear",
               "target_normalization_type": "range_linear",
               "target_normalization_parameters": {"dict": True},
               "targets_shape": (24, 24),
               "background_color": (0,),
               "validation_ratio": 0.15},
    "snapshotter": {"prefix": "kanji"},
    "weights_plotter": {"limit": 16},
    "layers": [{"name": "fc_tanh1",
                "type": "all2all_tanh",
                "->": {"output_sample_shape": 250},
                "<-": {"learning_rate": 0.0001, "weights_decay": 0.00005}},
               {"name": "fc_tanh2",
                "type": "all2all_tanh",
                "->": {"output_sample_shape": 250},
                "<-": {"learning_rate": 0.0001, "weights_decay": 0.00005}},
               {"name": "fc_tanh3",
                "type": "all2all_tanh",
                "->": {"output_sample_shape": (24, 24)},
                "<-": {"learning_rate": 0.0001, "weights_decay": 0.00005}}]})
