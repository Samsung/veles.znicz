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

Configuration file for cifar (Self-constructing Model).
Model - Network in network. (http://arxiv.org/abs/1312.4400)

Converged to 10.03% errors with some seed,
and below 11% in most cases.

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


train_dir = os.path.join(root.common.dirs.datasets, "cifar-10-batches-py")
validation_dir = os.path.join(root.common.dirs.datasets,
                              "cifar-10-batches-py/test_batch")

root.common.engine.precision_type = "float"
root.common.engine.precision_level = 1

base_lr = 0.1
momentum = 0.9
weights_decay = 0.0001

root.cifar.lr_adjuster.lr_parameters = {
    "lrs_with_lengths": [(1, 100000), (0.1, 100000), (0.01, 100000000)]}
root.cifar.lr_adjuster.bias_lr_parameters = {
    "lrs_with_lengths": [(1, 100000), (0.1, 100000), (0.01, 100000000)]}

root.cifar.update({
    "loader_name": "cifar_loader",
    "downloader": {
        "url": "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "directory": root.common.dirs.datasets,
        "files": ["cifar-10-batches-py"]},
    "decision": {"fail_iterations": 250, "max_epochs": 1000000000},
    "lr_adjuster": {"do": True, "lr_policy_name": "arbitrary_step",
                    "bias_lr_policy_name": "arbitrary_step"},
    "snapshotter": {"prefix": "cifar_nin", "interval": 1},
    "loss_function": "softmax",
    "add_plotters": True,
    "image_saver": {"do": False,
                    "out_dirs":
                    [os.path.join(root.common.dirs.cache, "tmp/test"),
                     os.path.join(root.common.dirs.cache, "tmp/validation"),
                     os.path.join(root.common.dirs.cache, "tmp/train")]},
    "loader": {"minibatch_size": 128,
               "normalization_type": "internal_mean",
               "add_sobel": False,
               "shuffle_limit": 2000000000,
               "force_numpy": False},
    "softmax": {"error_function_avr": True},
    "weights_plotter": {"limit": 256},
    "similar_weights_plotter": {"form_threshold": 1.1, "peak_threshold": 0.5,
                                "magnitude_threshold": 0.65},
    "layers": [{"name": "conv1",
                "type": "conv",
                "->": {"n_kernels": 192, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2),
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },
               {"name": "relu1", "type": "activation_str"},

               {"name": "conv2",
                "type": "conv",
                "->": {"n_kernels": 160, "kx": 1, "ky": 1,
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       # "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },

               {"name": "relu2", "type": "activation_str"},

               {"name": "conv3",
                "type": "conv",
                "->": {"n_kernels": 96, "kx": 1, "ky": 1,
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },
               {"name": "relu3", "type": "activation_str"},

               {"name": "pool3", "type": "max_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"name": "drop3", "type": "dropout", "dropout_ratio": 0.5},

               {"name": "conv4",
                "type": "conv",
                "->": {"n_kernels": 192, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2),
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },
               {"name": "relu4", "type": "activation_str"},

               {"name": "conv5",
                "type": "conv",
                "->": {"n_kernels": 192, "kx": 1, "ky": 1,
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },
               {"name": "relu5", "type": "activation_str"},

               {"name": "conv6",
                "type": "conv",
                "->": {"n_kernels": 192, "kx": 1, "ky": 1,
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },
               {"name": "relu6", "type": "activation_str"},

               {"name": "pool6", "type": "avg_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"name": "drop6", "type": "dropout", "dropout_ratio": 0.5},

               {"name": "conv7",
                "type": "conv",
                "->": {"n_kernels": 192, "kx": 3, "ky": 3,
                       "padding": (1, 1, 1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },
               {"name": "relu7", "type": "activation_str"},

               {"name": "conv8",
                "type": "conv",
                "->": {"n_kernels": 192, "kx": 1, "ky": 1,
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },
               {"name": "relu8", "type": "activation_str"},

               {"name": "conv9",
                "type": "conv",
                "->": {"n_kernels": 10, "kx": 1, "ky": 1,
                       "weights_filling": "gaussian", "weights_stddev": 0.05,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr * 0.1,
                       "learning_rate_bias": base_lr * 0.1,
                       "weights_decay": weights_decay,
                       "weights_decay_bias": 0,
                       "factor_ortho": 0.001,
                       "gradient_moment": momentum,
                       "gradient_moment_bias": momentum},
                },
               {"name": "relu9", "type": "activation_str"},

               {"name": "pool9", "type": "avg_pooling",
                "->": {"kx": 8, "ky": 8, "sliding": (1, 1)}},

               {"name": "fc_softmax4", "type": "softmax",
                "->": {"output_sample_shape": 10,
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                       "weights_decay": 1.0, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}],
    "data_paths": {"train": train_dir, "validation": validation_dir}})
