#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for Mnist. Model - fully-connected Neural Network with MSE
loss function with target encoded as 7 points.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters
root.mnist7.update({
    "decision": {"fail_iterations": 25, "max_epochs": 1000000},
    "snapshotter": {"prefix": "mnist7"},
    "loader": {"minibatch_size": 60, "on_device": True},
    "weights_plotter": {"limit": 25},
    "learning_rate": 0.0001,
    "weights_decay": 0.00005,
    "layers": [100, 100, 7]})
