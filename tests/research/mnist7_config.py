#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters
root.mnist7.update({
    "decision": {"fail_iterations": 25,
                 "snapshot_prefix": "mnist7"},
    "loader": {"minibatch_size": 60},
    "weights_plotter": {"limit": 25},
    "learning_rate": 0.0000016,
    "weights_decay": 0.00005,
    "layers": [100, 100, 7]})
