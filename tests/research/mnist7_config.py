#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters
root.update = {"decision": {"fail_iterations": 25,
                            "snapshot_prefix": "mnist7"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 25},
               "mnist7": {"learning_rate": 0.0000016,
                          "weights_decay": 0.00005,
                          "layers": [100, 100, 7]}}
