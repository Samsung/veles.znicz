#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on May 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root


# optional parameters

root.update = {"all2all_relu": {"weights_filling": "uniform",
                                "weights_magnitude": 0.05},
               "conv_relu1":  {"weights_filling": "gaussian",
                               "weights_stddev": 0.001},
               "conv_relu2":  {"weights_filling": "gaussian",
                               "weights_stddev": 0.001},
               "decision": {"fail_iterations": 100,
                            "snapshot_prefix": "lines"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 32},
               "lines": {"global_alpha": 0.01,
                         "global_lambda": 0.0,
                         "layers":
                         [{"type": "conv_relu1", "n_kernels": 32,
                              "kx": 7, "ky": 7, "padding": (2, 2, 2, 2)},
                             {"type": "max_pooling",
                              "kx": 3, "ky": 3, "sliding": (2, 2)},
                             {"type": "conv_relu2", "n_kernels": 16,
                              "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                             {"type": "avg_pooling",
                              "kx": 3, "ky": 3, "sliding": (2, 2)},
                             {"type": "relu", "layers": 16},
                             {"type": "softmax", "layers": 4}],
                            "snapshot": ""},
               "softmax": {"weights_filling": "uniform",
                           "weights_magnitude": 0.05}}
"""
[{"type": "conv_relu1", "n_kernels": 32,
                           "kx": 5, "ky": 5, "sliding": (4, 4),
                           "padding": (0, 0, 0, 0)},
                          {"type": "max_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)},
                          {"type": "conv_relu2", "n_kernels": 32,
                           "kx": 5, "ky": 5, "sliding": (4, 4),
                           "padding": (0, 0, 0, 0)},
                          {"type": "max_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)},
                          {"type": "relu", "layers": 32},
                          {"type": "softmax", "layers": 4}]},
"""
