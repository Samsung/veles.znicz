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
root.common.precision_type = "float"
#root.common.precision_level = 1
root.update = {
    "decision": {"fail_iterations": 1000000,
                 "use_dynamic_alpha": False,
                 "do_export_weights": True},
    "snapshotter": {"prefix": "imagenet"},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir,
                                  "tmp_imagenet/test"),
                     os.path.join(root.common.cache_dir,
                                  "tmp_imagenet/validation"),
                     os.path.join(root.common.cache_dir,
                                  "tmp_imagenet/train")]},
    "loader": {"year": "temp",
               "series": "img",
               "minibatch_size": 60},
    "weights_plotter": {"limit": 64},
    "imagenet": {"layers":
                 [{"type": "conv_relu", "n_kernels": 96,
                   "kx": 11, "ky": 11, "padding": (0, 0, 0, 0),
                   "sliding": (4, 4),
                   "learning_rate": 0.01, "learning_rate_bias": 0.02,
                   "weights_decay": 0.0005, "weights_decay_bias": 0.0,
                   "weights_filling": "gaussian",
                   "bias_filling": "constant", "bias_stddev": 0,
                   "weights_stddev": 0.01, "gradient_moment": 0.9,
                   "gradient_moment_bias": 0.9},
                  {"type": "max_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                  {"type": "norm", "alpha": 0.0001,
                   "beta": 0.75, "n": 5, "k": 1},

                  {"type": "conv_relu", "n_kernels": 256,
                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                   "learning_rate": 0.01, "learning_rate_bias": 0.02,
                   "weights_decay": 0.0005, "weights_decay_bias": 0.0,
                   "bias_filling": "constant", "bias_stddev": 1,
                   "gradient_moment": 0.9,
                   "gradient_moment_bias": 0.9,
                   "weights_filling": "gaussian",
                   "weights_stddev": 0.01},
                  {"type": "max_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                  {"type": "norm", "alpha": 0.0001,
                   "beta": 0.75, "n": 5, "k": 1},

                  {"type": "conv_relu", "n_kernels": 384,
                   "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                   "learning_rate": 0.01, "learning_rate_bias": 0.02,
                   "bias_filling": "constant", "bias_stddev": 0,
                   "weights_decay": 0.0005, "weights_decay_bias": 0.0,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9,
                   "weights_filling": "gaussian",
                   "weights_stddev": 0.01},

                  {"type": "conv_relu", "n_kernels": 384,
                   "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                   "learning_rate": 0.01, "learning_rate_bias": 0.02,
                   "bias_filling": "constant", "bias_stddev": 1,
                   "weights_decay": 0.0005, "weights_decay_bias": 0.0,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9,
                   "weights_filling": "gaussian",
                   "weights_stddev": 0.01},

                  {"type": "conv_relu", "n_kernels": 256,
                   "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                   "learning_rate": 0.01, "learning_rate_bias": 0.02,
                   "bias_filling": "constant", "bias_stddev": 1,
                   "weights_decay": 0.0005, "weights_decay_bias": 0.0,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9,
                   "weights_filling": "gaussian",
                   "weights_stddev": 0.01},

                  {"type": "max_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},

                  {"type": "all2all_relu", "output_shape": 4096,
                   "learning_rate": 0.01, "learning_rate_bias": 0.02,
                   "bias_filling": "constant", "bias_stddev": 1,
                   "weights_decay": 0.0005, "weights_decay_bias": 0.0,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9,
                   "weights_filling": "gaussian",
                   "weights_stddev": 0.005},

                  #{"type": "dropout", "dropout_ratio": 0.5},

                  {"type": "all2all_relu", "output_shape": 4096,
                   "weights_filling": "gaussian",
                   "weights_stddev": 0.005,
                   "learning_rate": 0.01, "learning_rate_bias": 0.02,
                   "bias_filling": "constant", "bias_stddev": 1,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9,
                   "weights_decay": 0.0005, "weights_decay_bias": 0.0},

                  #{"type": "dropout", "dropout_ratio": 0.5},

                  {"type": "softmax", "output_shape": 4,
                   "gradient_moment": 0.9, "gradient_moment_bias": 0.9,
                   "bias_filling": "constant", "bias_stddev": 0,
                   "learning_rate": 0.01, "learning_rate_bias": 0.02,
                   "weights_decay": 0.0005, "weights_decay_bias": 0.0,
                   "weights_filling": "gaussian",
                   "weights_stddev": 0.01}]}}

CACHED_DATA_FNME = os.path.join(IMAGENET_BASE_PATH, root.loader.year)
root.loader.names_labels_dir = os.path.join(
    CACHED_DATA_FNME, "original_labels_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
root.loader.count_samples_dir = os.path.join(
    CACHED_DATA_FNME, "count_samples_%s_%s_0.json" %
    (root.loader.year, root.loader.series))
root.loader.file_samples_dir = os.path.join(
    CACHED_DATA_FNME, "original_data_%s_%s_0.dat" %
    (root.loader.year, root.loader.series))
root.loader.matrixes_dir = os.path.join(
    CACHED_DATA_FNME, "matrixes_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
