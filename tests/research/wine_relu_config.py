#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for Wine.
Model - fully-connected Neural Network with SoftMax loss function with RELU
activation.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

root.common.update = {"plotters_disabled": True}

root.wine_relu.update({
    "decision": {"fail_iterations": 250, "max_epochs": 100000},
    "snapshotter": {"prefix": "wine_relu"},
    "loader": {"minibatch_size": 10, "on_device": True},
    "learning_rate": 0.03,
    "weights_decay": 0.0,
    "layers": [10, 3],
    "data_paths":
    os.path.join(root.common.veles_dir, "veles/znicz/samples/wine/wine.data")})
