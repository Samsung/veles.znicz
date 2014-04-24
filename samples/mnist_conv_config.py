#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root, Config

root.decision = Config()  # not necessary for execution (it will do it in real
root.loader = Config()  # time any way) but good for Eclipse editor

# optional parameters

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "mnist_conv"},
               "loader": {"minibatch_maxsize": 540},
               "weights_plotter": {"limit": 64},
               "mnist_conv": {"global_alpha": 0.005,
                              "global_lambda": 0.00005,
                              "layers":
                              [{"type": "conv", "n_kernels": 25,
                                "kx": 9, "ky": 9}, 100, 10]}}
