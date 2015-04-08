# -*-coding: utf-8 -*-
"""
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

"""
train = os.path.join(root.common.test_dataset_root,
                     "Lines/LINES_10_500_NOISY_min_valid/learning")
valid = os.path.join(root.common.test_dataset_root,
                     "Lines/LINES_10_500_NOISY_min_valid/test")
"""
train = os.path.join(root.common.test_dataset_root,
                     "Lines/lines_min/learn")
valid = os.path.join(root.common.test_dataset_root,
                     "Lines/lines_min/test")

root.lines.update({
    "loader_name": "full_batch_auto_label_file_image",
    "loss_function": "softmax",
    "decision": {"fail_iterations": 100,
                 "max_epochs": numpy.iinfo(numpy.uint32).max},
    "snapshotter": {"prefix": "lines", "interval": 1, "time_interval": 0},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 12, "force_numpy": False,
               "color_space": "RGB", "file_subtypes": ["jpeg"],
               "normalization_type": "mean_disp",
               "train_paths": [train], "validation_paths": [valid]},
    "weights_plotter": {"limit": 32},
    "layers": [{"type": "conv_relu",
                "->": {"n_kernels": 32, "kx": 11, "ky": 11, "sliding": (4, 4),
                       "weights_filling": "gaussian", "weights_stddev": 0.001,
                       "bias_filling": "gaussian", "bias_stddev": 0.001},
                "<-": {"learning_rate": 0.003,
                       "weights_decay": 0.0, "gradient_moment": 0.9},
                },
               {"type": "max_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},
               {"type": "all2all_relu",
                "->": {"output_sample_shape": 32, "weights_filling": "uniform",
                       "weights_stddev": 0.05, "bias_filling": "uniform",
                       "bias_stddev": 0.05},
                "<-": {"learning_rate": 0.001, "weights_decay": 0.0,
                       "gradient_moment": 0.9},
                },
               {"type": "softmax",
                "->": {"output_sample_shape": 4,
                       "weights_filling": "uniform",
                       "weights_stddev": 0.05, "bias_filling": "uniform",
                       "bias_stddev": 0.05},
                "<-": {"learning_rate": 0.001, "weights_decay": 0.0,
                       "gradient_moment": 0.9}, }]})
