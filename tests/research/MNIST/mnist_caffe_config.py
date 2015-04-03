#!/usr/bin/python3 -O
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Mart 21, 2014

Configuration file for Mnist. Configuration parameters just like
in CAFFE. Model - convolutional neural network.

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


from veles.config import root


root.mnistr.lr_adjuster.lr_parameters = {
    "base_lr": 0.01, "gamma": 0.0001, "pow_ratio": 0.75}
root.mnistr.lr_adjuster.bias_lr_parameters = {
    "base_lr": 0.01, "gamma": 0.0001, "pow_ratio": 0.75}

root.mnistr.update({
    "loss_function": "softmax",
    "loader_name": "mnist_loader",
    "lr_adjuster": {"do": True, "lr_policy_name": "inv",
                    "bias_lr_policy_name": "inv"},
    "decision": {"fail_iterations": 100, "max_epochs": 10000},
    "snapshotter": {"prefix": "mnist_caffe",
                    "time_interval": 0, "compress": ""},
    "loader": {"minibatch_size": 64, "force_cpu": False,
               "normalization_type": "linear"},
    "weights_plotter": {"limit": 64},
    "layers": [{"type": "conv",
                "->": {"n_kernels": 20, "kx": 5, "ky": 5,
                       "sliding": (1, 1), "weights_filling": "uniform",
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.01, "learning_rate_bias": 0.02,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0,
                       "weights_decay": 0.0005, "weights_decay_bias": 0}},

               {"type": "max_pooling",
                "->": {"kx": 2, "ky": 2, "sliding": (2, 2)}},

               {"type": "conv",
                "->": {"n_kernels": 50, "kx": 5, "ky": 5,
                       "sliding": (1, 1), "weights_filling": "uniform",
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.01, "learning_rate_bias": 0.02,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0}},

               {"type": "max_pooling",
                "->": {"kx": 2, "ky": 2, "sliding": (2, 2)}},

               {"type": "all2all_relu",
                "->": {"output_sample_shape": 500,
                       "weights_filling": "uniform",
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.01, "learning_rate_bias": 0.02,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0}},

               {"type": "softmax",
                "->": {"output_sample_shape": 10, "weights_filling": "uniform",
                       "bias_filling": "constant"},
                "<-": {"learning_rate": 0.01, "learning_rate_bias": 0.02,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0}}]})
