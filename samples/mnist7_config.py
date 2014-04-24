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
root.update = {"decision": {"fail_iterations": 25,
                            "snapshot_prefix": "mnist7"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 25},
               "mnist7": {"global_alpha": 0.0001,
                          "global_lambda": 0.00005,
                          "layers": [100, 100, 7]}}
