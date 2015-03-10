#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for Wine.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root

# optional parameters

root.common.update = {"disable_plotting": True}

root.wine.update({
    "decision": {"fail_iterations": 200,
                 "max_epochs": 100},
    "snapshotter": {"prefix": "wine", "interval": 1, "time_interval": 0},
    "loader": {"minibatch_size": 10,
               "force_cpu": False},
    "learning_rate": 0.3,
    "weights_decay": 0.0,
    "layers": [8, 3]})
