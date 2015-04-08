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
Model - convolutional neural network. Configuration parameters just like
in CAFFE.

Converged to 17.26% errors with some seed,
and below 18% in most cases.

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

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.common.precision_type = "float"
root.common.precision_level = 1

root.cifar.lr_adjuster.lr_parameters = {
    "lrs_with_lengths": [(1, 60000), (0.1, 5000), (0.01, 100000000)]}
root.cifar.lr_adjuster.bias_lr_parameters = {
    "lrs_with_lengths": [(1, 60000), (0.1, 5000), (0.01, 100000000)]}

root.cifar.update({
    "loader_name": "cifar_loader",
    "decision": {"fail_iterations": 250, "max_epochs": 1000000000},
    "lr_adjuster": {"do": True, "lr_policy_name": "arbitrary_step",
                    "bias_lr_policy_name": "arbitrary_step"},
    "snapshotter": {"prefix": "cifar_caffe", "interval": 1},
    "loss_function": "softmax",
    "add_plotters": True,
    "image_saver": {"do": False,
                    "out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 100,
               "normalization_type": "internal_mean",
               "add_sobel": False,
               "shuffle_limit": 2000000000,
               "force_numpy": False},
    "softmax": {"error_function_avr": True},
    "weights_plotter": {"limit": 64},
    "similar_weights_plotter": {"form_threshold": 1.1, "peak_threshold": 0.5,
                                "magnitude_threshold": 0.65},
    "layers": [{"name": "conv1",
                "type": "conv",
                "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.0001,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                       "factor_ortho": 0.001, "gradient_moment": 0.9,
                       "gradient_moment_bias": 0.9},
                },
               {"name": "pool1", "type": "max_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"name": "relu1", "type": "activation_str"},

               {"name": "norm1", "type": "norm", "alpha": 0.00005,
                "beta": 0.75, "n": 3, "k": 1},

               {"name": "conv2", "type": "conv",
                "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                       "factor_ortho": 0.001, "gradient_moment": 0.9,
                       "gradient_moment_bias": 0.9}
                },
               {"name": "relu2", "type": "activation_str"},

               {"name": "pool2", "type": "avg_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"name": "norm2", "type": "norm",
                "alpha": 0.00005, "beta": 0.75, "n": 3, "k": 1},

               {"name": "conv3", "type": "conv",
                "->": {"n_kernels": 64, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2), "bias_stddev": 0,
                       "sliding": (1, 1), "weights_filling": "gaussian",
                       "weights_stddev": 0.01, "bias_filling": "constant"},
                "<-": {"learning_rate": 0.001,
                       "learning_rate_bias": 0.001, "weights_decay": 0.0005,
                       "weights_decay_bias": 0.0005, "factor_ortho": 0.001,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
                },
               {"name": "relu3", "type": "activation_str"},

               {"name": "pool3", "type": "avg_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"name": "a2asm4", "type": "softmax",
                "->": {"output_sample_shape": 10,
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                       "weights_decay": 1.0, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}],
    "data_paths": {"train": train_dir, "validation": validation_dir}})
