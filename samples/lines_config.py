#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on May 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train = "/data/veles/Lines/LINES_10_500_NOISY_min_valid/learning"
valid = "/data/veles/Lines/LINES_10_500_NOISY_min_valid/test"

root.model = "lines"

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "lines"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 32},
               "image_saver": {"out_dirs":
                                 [os.path.join(root.common.cache_dir,
                                               "tmp %s/test" % root.model),
                                  os.path.join(root.common.cache_dir,
                                               "tmp %s/validation" %
                                               root.model),
                                  os.path.join(root.common.cache_dir,
                                               "tmp %s/train" % root.model)]},
               "lines": {"learning_rate": 0.01,
                         "weights_decay": 0.0,
                         "layers":
                         [{"type": "conv_relu", "n_kernels": 6,
                           "kx": 11, "ky": 11,
                           "sliding": (4, 4),
                           #"padding": (2, 2, 2, 2),
                           "learning_rate": 0.03, "weights_decay": 0.0,
                           "gradient_moment": 0.9,
                           "weights_filling": "gaussian",
                           "weights_stddev": 0.001,
                           "bias_filling": "gaussian",
                           "bias_stddev": 0.001
                           },
                          {"type": "max_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)},
                          {"type": "conv_relu", "n_kernels": 16,
                           "kx": 7, "ky": 7,
                           "sliding": (4, 4),
                           #"padding": (2, 2, 2, 2),
                           "learning_rate": 0.01, "weights_decay": 0.0,
                           "gradient_moment": 0.9,
                           "weights_filling": "gaussian",
                           "bias_filling": "gaussian",
                           },
                          {"type": "avg_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)},
                          {"type": "all2all_relu", "output_shape": 16,
                           "learning_rate": 0.01, "weights_decay": 0.0,
                           "gradient_moment": 0.9,
                           "weights_filling": "uniform",
                           #"weights_stddev": 0.05,
                           "bias_filling": "uniform",
                           #"bias_stddev": 0.05
                           },
                          {"type": "softmax", "output_shape": 4,
                           "learning_rate": 0.01, "weights_decay": 0.0,
                           "gradient_moment": 0.9,
                           "weights_filling": "uniform",
                           #"weights_stddev": 0.05,
                           "bias_filling": "uniform",
                           #"bias_stddev": 0.05
                           }],
                         "path_for_load_data": {"validation": valid,
                                                "train": train}}}
