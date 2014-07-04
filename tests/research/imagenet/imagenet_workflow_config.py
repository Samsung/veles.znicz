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
    "decision": {"fail_iterations": 100000,
                 "use_dynamic_alpha": False,
                 "do_export_weights": True},
    "snapshotter": {"prefix": "imagenet"},
    "loader": {"year": "temp",
               "series": "img",
               "minibatch_size": 100},
    "weights_plotter": {"limit": 64},
    "imagenet": {"layers":
                 [{"type": "conv_relu", "n_kernels": 96,
                   "kx": 11, "ky": 11, "padding": (0, 0, 0, 0),
                   "sliding": (4, 4), "weights_filling": "gaussian",
                   "bias_filling": "constant", "bias_stddev": 0,
                   "weights_stddev": 0.01, "learning_rate": 0.001,
                   "learning_rate_bias": 0.002, "weights_decay": 0.004,
                   "weights_decay_bias": 0.004, "gradient_moment": 0.9,
                   "gradient_moment_bias": 0.9},
                  {"type": "max_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                  {"type": "norm", "alpha": 0.00005, "beta": 0.75, "n": 3},

                  {"type": "conv_relu", "n_kernels": 256,
                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                   "sliding": (1, 1),
                   "weights_filling": "gaussian", "bias_filling": "constant",
                   "bias_stddev": 0, "weights_stddev": 0.01,
                   "learning_rate": 0.001, "learning_rate_bias": 0.002,
                   "weights_decay": 0.004, "weights_decay_bias": 0.004,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
                  {"type": "max_pooling", "kx": 3, "ky": 3, "sliding": (2, 2)},

                  {"type": "conv", "n_kernels": 384,
                   "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                   "sliding": (1, 1), "weights_filling": "gaussian",
                   "weights_stddev": 0.01, "bias_filling": "constant",
                   "bias_stddev": 0,
                   "learning_rate": 0.001, "learning_rate_bias": 0.002,
                   "weights_decay": 0.004, "weights_decay_bias": 0.004,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9},

                  {"type": "conv", "n_kernels": 384,
                   "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                   "sliding": (1, 1), "weights_filling": "gaussian",
                   "weights_stddev": 0.01, "bias_filling": "constant",
                   "bias_stddev": 0,
                   "learning_rate": 0.001, "learning_rate_bias": 0.002,
                   "weights_decay": 0.004, "weights_decay_bias": 0.004,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9},

                  {"type": "conv_relu", "n_kernels": 256,
                   "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                   "sliding": (1, 1), "weights_filling": "gaussian",
                   "weights_stddev": 0.01, "bias_filling": "constant",
                   "bias_stddev": 0,
                   "learning_rate": 0.001, "learning_rate_bias": 0.002,
                   "weights_decay": 0.004, "weights_decay_bias": 0.004,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
                  {"type": "max_pooling", "kx": 3, "ky": 3, "sliding": (2, 2)},

                  {"type": "all2all_relu", "output_shape": 4096,
                   "weights_filling": "gaussian", "weights_stddev": 0.005,
                   "bias_filling": "constant", "bias_stddev": 0,
                   "learning_rate": 0.001,
                   "learning_rate_bias": 0.002, "weights_decay": 0.004,
                   "weights_decay_bias": 0.004, "gradient_moment": 0.9,
                   "gradient_moment_bias": 0.9},

                  {"type": "dropout", "dropout_ratio": 0.5},

                  {"type": "softmax", "output_shape": 1000,
                   "weights_filling": "gaussian",
                   "weights_stddev": 0.01, "bias_filling": "constant",
                   "bias_stddev": 0,
                   "learning_rate": 0.001, "learning_rate_bias": 0.002,
                   "weights_decay": 0.004, "weights_decay_bias": 0.004,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9}]}}

CACHED_DATA_FNME = os.path.join(IMAGENET_BASE_PATH, root.loader.year)
root.loader.names_labels_dir = os.path.join(
    CACHED_DATA_FNME, "names_labels_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
root.loader.count_samples_dir = os.path.join(
    CACHED_DATA_FNME, "count_samples_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
