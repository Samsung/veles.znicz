#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on May 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root


# optional parameters

root.update = {"conv_relu":  {  #"weights_filling": "uniform",
                                #"weights_magnitude": 0.000001
                                "weights_filling": "gaussian",
                                "weights_stddev": 0.000001},
               "decision": {"fail_iterations": 100,
                            "snapshot_prefix": "lines"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 32},
               "lines": {"global_alpha": 0.02,
                         "global_lambda": 0.1,
                         "layers":
                         [{"type": "conv_relu", "n_kernels": 4,
                           "kx": 11, "ky": 11, "sliding": (4, 4),
                           "padding": (0, 0, 0, 0)},
                          {"type": "max_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)},
                          {"type": "softmax", "layers": 4}]},
               "softmax": {"weights_filling": "gaussian",
                           "weights_stddev": 0.01}}
