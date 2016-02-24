# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 20, 2014

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

LR = 0.01
WD = 0.0005
ORTHO = 0.001
GM = 0.9
L1_VS_L2 = 0.0

FILLING = "gaussian"

root.common.engine.precision_type = "float"
root.common.engine.precision_level = 0
root.common.engine.backend = "cuda"

root.imagenet.root_name = "imagenet"
root.imagenet.series = "img"
root.imagenet.root_path = os.path.join(
    root.common.dirs.datasets, "AlexNet", "%s" % root.imagenet.root_name)

root.imagenet.lr_adjuster.lr_parameters = {
    "lrs_with_lengths":
    [(1, 200000), (0.1, 200000), (0.01, 200000), (0.001, 100000000)]}
root.imagenet.lr_adjuster.bias_lr_parameters = {
    "lrs_with_lengths":
    [(1, 200000), (0.1, 200000), (0.01, 200000), (0.001, 100000000)]}

root.imagenet.loader.update({
    "sx": 256,
    "sy": 256,
    "crop_size_sx": 224,
    "crop_size_sy": 224,
    "mirror": True,
    "channels": 3,
    "minibatch_size": 64,
    "normalization_type": "none",
    "shuffle_limit": 1,
    "original_labels_filename":
    os.path.join(
        root.imagenet.root_path,
        "original_labels_%s_%s.pickle"
        % (root.imagenet.root_name, root.imagenet.series)),
    "samples_filename":
    os.path.join(
        root.imagenet.root_path,
        "original_data_%s_%s.dat"
        % (root.imagenet.root_name, root.imagenet.series)),
    "matrixes_filename":
    os.path.join(
        root.imagenet.root_path,
        "matrixes_%s_%s.pickle"
        % (root.imagenet.root_name, root.imagenet.series)),
    "count_samples_filename":
    os.path.join(
        root.imagenet.root_path,
        "count_samples_%s_%s.json"
        % (root.imagenet.root_name, root.imagenet.series)),
    "class_keys_path": os.path.join(
        root.imagenet.root_path,
        "class_keys_%s_%s.json" %
        (root.imagenet.root_name, root.imagenet.series))
})

root.imagenet.update({
    "decision": {"fail_iterations": 10000,
                 "max_epochs": 10000},
    "snapshotter": {"prefix": "imagenet", "interval": 1, "time_interval": 0,
                    "directory": os.path.join(root.common.dirs.datasets,
                                              "AlexNet/snapshots")},
    "add_plotters": True,
    "loss_function": "softmax",
    "lr_adjuster": {"lr_policy_name": "arbitrary_step",
                    "bias_lr_policy_name": "arbitrary_step"},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.dirs.datasets,
                                  "AlexNet/image_saver/test"),
                     os.path.join(root.common.dirs.datasets,
                                  "AlexNet/image_saver/validation"),
                     os.path.join(root.common.dirs.datasets,
                                  "AlexNet/image_saver/train")]},
    "loader_name": "imagenet_pickle_loader",
    "weights_plotter": {"limit": 256, "split_channels": False},
    "layers":
    [{"name": "conv1",
      "type": "conv",
      "->": {"n_kernels": 96, "kx": 11, "ky": 11, "sliding": (4, 4),
             "weights_filling": FILLING,
             "weights_stddev": 0.01,
             "bias_filling": "constant", "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu1", "type": "activation_str"},
     {"name": "conv2",
      "type": "conv",
      "->": {"n_kernels": 96, "kx": 1, "ky": 1, "sliding": (1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.05,
             "bias_filling": "constant", "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu2", "type": "activation_str"},
     {"name": "conv3",
      "type": "conv",
      "->": {"n_kernels": 96, "kx": 1, "ky": 1, "sliding": (1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.05,
             "bias_filling": "constant", "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu3", "type": "activation_str"},
     {"name": "pool3", "type": "max_pooling",
      "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

     {"name": "conv4",
      "type": "conv",
      "->": {"n_kernels": 256, "kx": 5, "ky": 5, "sliding": (1, 1),
             "padding": (2, 2, 2, 2), "weights_filling": FILLING,
             "weights_stddev": 0.05, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu4", "type": "activation_str"},

     {"name": "conv5",
      "type": "conv",
      "->": {"n_kernels": 256, "kx": 1, "ky": 1, "sliding": (1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.05, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu5", "type": "activation_str"},

     {"name": "conv6",
      "type": "conv",
      "->": {"n_kernels": 256, "kx": 1, "ky": 1, "sliding": (1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.05, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu6", "type": "activation_str"},

     {"name": "pool6", "type": "max_pooling",
      "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

     {"name": "conv7",
      "type": "conv",
      "->": {"n_kernels": 384, "kx": 3, "ky": 3, "sliding": (1, 1),
             "padding": (1, 1, 1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.01, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu7", "type": "activation_str"},

     {"name": "conv8",
      "type": "conv",
      "->": {"n_kernels": 384, "kx": 1, "ky": 1, "sliding": (1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.05, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu8", "type": "activation_str"},

     {"name": "conv9",
      "type": "conv",
      "->": {"n_kernels": 384, "kx": 1, "ky": 1, "sliding": (1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.05, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu9", "type": "activation_str"},

     {"name": "pool9", "type": "max_pooling",
      "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

     {"name": "drop9", "type": "dropout", "dropout_ratio": 0.5},

     {"name": "conv10",
      "type": "conv",
      "->": {"n_kernels": 1024, "kx": 3, "ky": 3, "sliding": (1, 1),
             "padding": (1, 1, 1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.05, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu10", "type": "activation_str"},

     {"name": "conv11",
      "type": "conv",
      "->": {"n_kernels": 1024, "kx": 1, "ky": 1, "sliding": (1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.05, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu11", "type": "activation_str"},


     {"name": "conv12",
      "type": "conv",
      "->": {"n_kernels": 1000, "kx": 1, "ky": 1, "sliding": (1, 1),
             "weights_filling": FILLING,
             "weights_stddev": 0.01, "bias_filling": "constant",
             "bias_stddev": 0},
      "<-": {"learning_rate": LR,
             "learning_rate_bias": LR * 2,
             "weights_decay": WD, "weights_decay_bias": 0,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2
             }},
     {"name": "relu12", "type": "activation_str"},

     {"name": "pool12", "type": "avg_pooling",
      "->": {"kx": 6, "ky": 6, "sliding": (1, 1)}},

     {"name": "fc_softmax13",
      "type": "softmax",
      "->": {"output_sample_shape": 1000,
             "weights_filling": "gaussian",
             "weights_stddev": 0.01},
      "<-": {"learning_rate": 0.01,
             "learning_rate_bias": 0.01 * 2,
             "weights_decay": 0.0005, "weights_decay_bias": 0,
             "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}]})
