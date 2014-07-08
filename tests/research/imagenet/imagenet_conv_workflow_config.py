#!/usr/bin/python3.3 -O

"""
Created on July 4, 2014

Imagenet recognition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


IMAGENET_BASE_PATH = os.path.join(root.common.test_dataset_root,
                                  "imagenet")

root.update = {
    "decision": {"fail_iterations": 250},
    "snapshotter": {"prefix": "imagenet"},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 20},
    "weights_plotter": {"limit": 16},
    "imagenet": {"layers":
                 [{"type": "conv", "n_kernels": 32,
                   "kx": 11, "ky": 11, "padding": (2, 2, 2, 2),
                   "sliding": (1, 1),
                   "weights_filling": "gaussian", "weights_stddev": 0.0001,
                   "bias_filling": "constant", "bias_stddev": 0,
                   "learning_rate": 0.001, "learning_rate_bias": 0.002,
                   "weights_decay": 0.004, "weights_decay_bias": 0.004,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
                  {"type": "max_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                  {"type": "activation_str"},
                  {"type": "norm", "alpha": 0.00005, "beta": 0.75,
                   "n": 3, "k": 1},

                  {"type": "conv", "n_kernels": 32,
                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                   "sliding": (1, 1),
                   "weights_filling": "gaussian", "weights_stddev": 0.01,
                   "bias_filling": "constant", "bias_stddev": 0,
                   "learning_rate": 0.001, "learning_rate_bias": 0.002,
                   "weights_decay": 0.004, "weights_decay_bias": 0.004,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
                  {"type": "activation_str"},
                  {"type": "avg_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                  {"type": "norm", "alpha": 0.00005, "beta": 0.75,
                   "n": 3, "k": 1},

                  {"type": "all2all_relu", "output_shape": 32,
                   "weights_filling": "gaussian", "weights_stddev": 0.005,
                   "bias_filling": "constant", "bias_stddev": 0,
                   "learning_rate": 0.001,
                   "learning_rate_bias": 0.002, "weights_decay": 0.004,
                   "weights_decay_bias": 0.004, "gradient_moment": 0.9,
                   "gradient_moment_bias": 0.9},

                  {"type": "softmax", "output_shape": 4,
                   "weights_filling": "gaussian", "weights_stddev": 0.01,
                   "bias_filling": "constant", "bias_stddev": 0,
                   "learning_rate": 0.001, "learning_rate_bias": 0.002,
                   "weights_decay": 1.0, "weights_decay_bias": 0,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9}]}}

CACHED_DATA_FNME = os.path.join(IMAGENET_BASE_PATH, root.loader.year)
root.loader.names_labels_dir = os.path.join(
    CACHED_DATA_FNME, "names_labels_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
root.loader.count_samples_dir = os.path.join(
    CACHED_DATA_FNME, "count_samples_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
