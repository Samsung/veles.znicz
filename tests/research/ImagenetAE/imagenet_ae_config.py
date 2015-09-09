# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jule 18, 2014

Configuration file for imagenet_ae with stochastic pooling.
Number of classes - 200. (DET)

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


LR = 0.0002
WD = 0.0005
ORTHO = 0.001
GM = 0.9
L1_VS_L2 = 0.0

LRFT = 0.01
LRFTB = LRFT * 2

LRAA = 0.01
LRBAA = LRAA * 2
WDAA = 0.0005
ORTHOAA = 0.001
WDBAA = 0
GMAA = 0.9
GMBAA = GM

FILLING = "gaussian"
STDDEV_CONV = 0.01
STDDEV_AA = 0.005

root.common.engine.precision_type = "float"
root.imagenet_ae.model = "imagenet"
root.imagenet_ae.update({
    "decision_mse": {"fail_iterations": 50,
                     "max_epochs": 50},
    "decision_gd": {"fail_iterations": 50,
                    "max_epochs": 50},
    # number of epochs to add to max_epochs if model is training from snapshot
    # and another layer was not added:
    "decision": {"add_epochs": 50},
    "rollback": {"lr_plus": 1},
    "loader_name": "imagenet_ae_loader",
    "lr_adjuster": {"lr_policy_name": "arbitrary_step",
                    "bias_lr_policy_name": "arbitrary_step"},
    "weights_plotter": {"limit": 256, "split_channels": False},
    "loader": {"year": "imagenet",
               "series": "DET",
               "minibatch_size": 120,
               "path": os.path.join(root.common.dirs.datasets, "AlexNet"),
               "sx": 216,
               "sy": 216},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.dirs.cache,
                                  "tmp_imagenet/test"),
                     os.path.join(root.common.dirs.cache,
                                  "tmp_imagenet/validation"),
                     os.path.join(root.common.dirs.cache,
                                  "tmp_imagenet/train")]},
    "snapshotter": {"prefix": "imagenet_ae",
                    "compression": "gz",
                    "directory":
                    os.path.join(root.common.dirs.datasets,
                                 "imagenet/snapshots/DET/new")},
    "from_snapshot_add_layer": True,
    "fine_tuning_noise": 1.0e-6,
    "layers":
    [{"type": "ae_begin"},  # 216
     {"name": "conv1",
      "type": "conv",
      "->": {"n_kernels": 108, "kx": 9, "ky": 9, "sliding": (3, 3),
             "weights_filling": FILLING,
             "weights_stddev": STDDEV_CONV},
      "<-": {"learning_rate": LR, "learning_rate_ft": LRFT,
             "learning_rate_bias": LR, "learning_rate_ft_bias": LRFT,
             "weights_decay": WD, "weights_decay_bias": WD,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2}},
     {"name": "pool1",
      "type": "stochastic_abs_pooling",
      "->": {"kx": 3, "ky": 3, "sliding": (3, 3)}},  # 69
     {"type": "ae_end"},

     {"name": "mul1", "type": "activation_mul"},

     {"type": "ae_begin"},  # 23
     {"name": "conv2",
      "type": "conv",
      "->": {"n_kernels": 192, "kx": 5, "ky": 5, "sliding": (1, 1),
             "padding": (2, 2, 2, 2), "weights_filling": FILLING,
             "weights_stddev": STDDEV_CONV},
      "<-": {"learning_rate": LR, "learning_rate_ft": LRFT,
             "learning_rate_bias": LR, "learning_rate_ft_bias": LRFT,
             "weights_decay": WD, "weights_decay_bias": WD,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2}},
     {"name": "pool2",
      "type": "stochastic_abs_pooling",
      "->": {"padding": (2, 2, 2, 2), "kx": 2, "ky": 2, "sliding": (2, 2)}},
     {"type": "ae_end"},

     {"name": "mul2", "type": "activation_mul"},

     {"type": "ae_begin"},  # 13
     {"name": "conv3",
      "type": "conv",
      "->": {"n_kernels": 224, "kx": 5, "ky": 5, "sliding": (1, 1),
             "padding": (2, 2, 2, 2), "weights_filling": FILLING,
             "weights_stddev": STDDEV_CONV},
      "<-": {"learning_rate": LR, "learning_rate_ft": LRFT,
             "learning_rate_bias": LR, "learning_rate_ft_bias": LRFT,
             "weights_decay": WD, "weights_decay_bias": WD,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2}},
     {"type": "ae_end"},

     {"name": "mul3", "type": "activation_mul"},

     {"type": "ae_begin"},  # 13
     {"name": "conv4",
      "type": "conv",
      "->": {"n_kernels": 256, "kx": 3, "ky": 3, "sliding": (1, 1),
             "padding": (1, 1, 1, 1), "weights_filling": FILLING,
             "weights_stddev": STDDEV_CONV},
      "<-": {"learning_rate": LR, "learning_rate_ft": LRFT,
             "learning_rate_bias": LR, "learning_rate_ft_bias": LRFT,
             "weights_decay": WD, "weights_decay_bias": WD,
             "factor_ortho": ORTHO,
             "gradient_moment": GM, "gradient_moment_bias": GM,
             "l1_vs_l2": L1_VS_L2}},
     {"name": "pool4", "type": "stochastic_abs_pooling",
      "->": {"kx": 2, "ky": 2, "sliding": (2, 2)}},  # 6
     {"type": "ae_end"},

     {"name": "mul4", "type": "activation_mul"},

     {"name": "fc_tanh5",
      "type": "all2all_tanh",
      "->": {"output_sample_shape": 4096, "weights_filling": "gaussian",
             "bias_filling": "constant", "weights_stddev": STDDEV_AA,
             "bias_stddev": 0.1},
      "<-": {"learning_rate": LRAA, "learning_rate_bias": LRBAA,
             "learning_rate_ft": LRFT, "learning_rate_ft_bias": LRFTB,
             "weights_decay": WDAA, "weights_decay_bias": WDBAA,
             "factor_ortho": ORTHOAA, "l1_vs_l2": L1_VS_L2,
             "gradient_moment": GMAA, "gradient_moment_bias": GMBAA}},
     {"name": "drop5", "type": "dropout", "dropout_ratio": 0.5},
     {"name": "fc_tanh6",
      "type": "all2all_tanh",
      "->": {"output_sample_shape": 4096,
             "weights_filling": "gaussian", "weights_stddev": STDDEV_AA,
             "bias_filling": "constant", "bias_stddev": 0.1},
      "<-": {"learning_rate": LRAA, "learning_rate_bias": LRBAA,
             "learning_rate_ft": LRFT, "learning_rate_ft_bias": LRFTB,
             "weights_decay": WDAA, "weights_decay_bias": WDBAA,
             "factor_ortho": ORTHOAA, "l1_vs_l2": L1_VS_L2,
             "gradient_moment": GMAA, "gradient_moment_bias": GMBAA}},
     {"name": "drop6", "type": "dropout", "dropout_ratio": 0.5},
     {"name": "fc_softmax7",
      "type": "softmax",
      "->": {"output_sample_shape": 200,
             "weights_filling": "gaussian", "weights_stddev": 0.01,
             "bias_filling": "constant", "bias_stddev": 0},
      "<-": {"learning_rate": LRAA, "learning_rate_bias": LRBAA,
             "learning_rate_ft": LRFT, "learning_rate_ft_bias": LRFTB,
             "weights_decay": WDAA, "weights_decay_bias": WDBAA,
             "gradient_moment": GMAA, "gradient_moment_bias": GMBAA,
             "l1_vs_l2": L1_VS_L2}}]})

root.imagenet_ae.lr_adjuster.lr_parameters = {
    "lrs_with_lengths":
    [(1, 100000), (0.1, 100000), (0.1, 100000), (0.01, 100000000)]}
root.imagenet_ae.lr_adjuster.bias_lr_parameters = {
    "lrs_with_lengths":
    [(1, 100000), (0.1, 100000), (0.1, 100000), (0.01, 100000000)]}

imagenet_base_path = root.imagenet_ae.loader.path
root.imagenet_ae.snapshotter.prefix = (
    "imagenet_ae_%s" % root.imagenet_ae.loader.year)
imagenet_data_path = os.path.join(
    imagenet_base_path, str(root.imagenet_ae.loader.year))
root.imagenet_ae.loader.names_labels_filename = os.path.join(
    imagenet_data_path, "original_labels_%s_%s.pickle" %
    (root.imagenet_ae.loader.year, root.imagenet_ae.loader.series))
root.imagenet_ae.loader.count_samples_filename = os.path.join(
    imagenet_data_path, "count_samples_%s_%s.json" %
    (root.imagenet_ae.loader.year, root.imagenet_ae.loader.series))
root.imagenet_ae.loader.samples_filename = os.path.join(
    imagenet_data_path, "original_data_%s_%s.dat" %
    (root.imagenet_ae.loader.year, root.imagenet_ae.loader.series))
root.imagenet_ae.loader.matrixes_filename = os.path.join(
    imagenet_data_path, "matrixes_%s_%s.pickle" %
    (root.imagenet_ae.loader.year, root.imagenet_ae.loader.series))
