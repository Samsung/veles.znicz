#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on May 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train = "/data/veles/Lines/Grid/learn"
valid = "/data/veles/Lines/Grid/test"

root.model = "grid"

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "lines"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 32},
               "image_saver": {"out_dirs":
                               [os.path.join(root.common.cache_dir,
                                             "tmp %s/test" % root.model),
                                os.path.join(root.common.cache_dir,
                                             "tmp %s/validation" % root.model),
                                os.path.join(root.common.cache_dir,
                                             "tmp %s/train" % root.model)]},
               "lines": {"learning_rate": 0.01,
                         "weights_decay": 0.0,
                         "layers":
                         [{"type": "conv_relu", "n_kernels": 32,
                           "kx": 13, "ky": 13,
                           "sliding": (2, 2),
                           "padding": (1, 1, 1, 1),
                           "learning_rate": 0.01,
                           "learning_rate_bias": 0.02,
                           "gradient_moment": 0.9,
                           "weights_filling": "gaussian",
                           "weights_stddev": 0.0001,
                           "bias_filling": "constant",
                           "bias_stddev": 0.0001,
                           "weights_decay": 0.0,
                           "weights_decay_bias": 0.0},
                          {"type": "max_pooling",
                           "kx": 5, "ky": 5, "sliding": (2, 2)},
                          {"type": "avg_pooling",
                           "kx": 5, "ky": 5, "sliding": (2, 2)},
                          {"type": "norm",
                           "alpha": 0.00005, "beta": 0.75, "n": 3},
                          {"type": "conv_relu", "n_kernels": 32,
                           "kx": 7, "ky": 7,
                           "sliding": (1, 1),
                           "padding": (2, 2, 2, 2),
                           "learning_rate": 0.01,
                           "learning_rate_bias": 0.02,
                           "gradient_moment": 0.9,
                           "weights_filling": "gaussian",
                           "weights_stddev": 0.01,
                           "bias_filling": "constant",
                           "bias_stddev": 0.01,
                           "weights_decay": 0.0,
                           "weights_decay_bias": 0.0},
                          {"type": "avg_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)},
                          {"type": "norm",
                           "alpha": 0.00005, "beta": 0.75, "n": 3},
                          {"type": "softmax", "output_shape": 6,
                           "gradient_moment": 0.9,
                           "weights_filling": "uniform",
                           "weights_stddev": 0.05,
                           "bias_filling": "constant", "bias_stddev": 0.05,
                           "learning_rate": 0.01, "learning_rate_bias": 0.02,
                           "weights_decay": 1, "weights_decay_bias": 0}],
                         "path_for_load_data": {"validation": valid,
                                                "train": train}}}
