#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on May 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root


# optional parameters

root.update = {"all2all_relu": {"weights_filling": "uniform",
                                "weights_stddev": 0.05},
               "conv_relu":  {"weights_filling": "gaussian",
                              "weights_stddev": 0.0001},
               "decision": {"fail_iterations": 100,
                            "snapshot_prefix": "lines"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 32},
               "lines": {"learning_rate": 0.01,
                         "weights_decay": 0.0,
                         "layers":
                         [{"type": "conv_relu", "n_kernels": 32,
                           "kx": 7, "ky": 7, "padding": (2, 2, 2, 2)},
                          {"type": "max_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)},
                          {"type": "conv_relu", "n_kernels": 16,
                           "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                          {"type": "all2all_relu", "layers": 100},
                          {"type": "softmax", "layers": 4}],
                         "snapshot": ""},
               "softmax": {"weights_filling": "uniform",
                           "weights_stddev": 0.05}}
