# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Mart 21, 2014

Configuration file for cifar (Self-constructing Model).
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


train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.cifar.update({
    "loader_name": "cifar_loader",
    "decision": {"fail_iterations": 1000, "max_epochs": 1000000000},
    "lr_adjuster": {"do": False},
    "snapshotter": {"prefix": "cifar", "interval": 1},
    "loss_function": "softmax",
    "add_plotters": True,
    "image_saver": {"do": False,
                    "out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 81, "force_cpu": False,
               "normalization_type": "linear"},
    "accumulator": {"n_bars": 30},
    "weights_plotter": {"limit": 25},
    "similar_weights_plotter": {"form_threshold": 1.1, "peak_threshold": 0.5,
                                "magnitude_threshold": 0.65},
    "layers": [{"type": "all2all",
                "->": {"output_sample_shape": 486},
                "<-": {"learning_rate": 0.0005, "weights_decay": 0.0}},
               {"type": "activation_sincos"},
               {"type": "all2all",
                "->": {"output_sample_shape": 486},
                "<-": {"learning_rate": 0.0005, "weights_decay": 0.0}},
               {"type": "activation_sincos"},
               {"type": "softmax",
                "->": {"output_sample_shape": 10},
                "<-": {"learning_rate": 0.0005, "weights_decay": 0.0}}],
    "data_paths": {"train": train_dir, "validation": validation_dir}})
