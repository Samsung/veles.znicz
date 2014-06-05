#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on May 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train = os.path.join(root.common.test_dataset_root,
                     "Lines/LINES_10_500_NOISY_min_valid/learning")
valid = os.path.join(root.common.test_dataset_root,
                     "Lines/LINES_10_500_NOISY_min_valid/test")


root.model = "lines"

root.update = {"accumulator": {"bars": 30, "squash": True},
               "decision": {"fail_iterations": 100},
               "snapshotter": {"prefix": "lines"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 32},
               "lines": {"learning_rate": 0.0001,
                         "weights_decay": 0.0,
                         "layers":
                         [{"type": "conv_relu", "n_kernels": 32,
                           "kx": 11, "ky": 11,
                           "sliding": (4, 4),
                           "learning_rate": 0.0003, "weights_decay": 0.0,
                           "gradient_moment": 0.9,
                           "weights_filling": "gaussian",
                           "weights_stddev": 0.001,
                           "bias_filling": "gaussian",
                           "bias_stddev": 0.001
                           },
                          {"type": "max_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)},
                          {"type": "all2all_relu", "output_shape": 32,
                           "learning_rate": 0.0001, "weights_decay": 0.0,
                           "gradient_moment": 0.9,
                           "weights_filling": "uniform",
                           "weights_stddev": 0.05,
                           "bias_filling": "uniform",
                           "bias_stddev": 0.05
                           },
                          {"type": "softmax", "output_shape": 4,
                           "learning_rate": 0.0001, "weights_decay": 0.0,
                           "gradient_moment": 0.9,
                           "weights_filling": "uniform",
                           "weights_stddev": 0.05,
                           "bias_filling": "uniform",
                           "bias_stddev": 0.05
                           }],
                         "path_for_load_data": {"validation": valid,
                                                "train": train}}}
