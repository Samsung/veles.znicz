#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "mnist_conv"},
               "loader": {"minibatch_maxsize": 540},
               "weights_plotter": {"limit": 64},
               "mnist_conv": {"learning_rate": 0.005,
                              "weights_decay": 0.00005,
                              "layers":
                              [{"type": "conv", "n_kernels": 25,
                                "kx": 9, "ky": 9}, 100, 10]}}
