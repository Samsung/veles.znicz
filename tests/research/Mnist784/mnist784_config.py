#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for Mnist. Model - fully-connected
Neural Network with MSE loss function with target encoded as ideal image
(784 points).

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


root.mnist784.update({
    "decision": {"fail_iterations": 100, "max_epochs": 100000},
    "snapshotter": {"prefix": "mnist_784"},
    "loader": {"minibatch_size": 100, "on_device": True},
    "weights_plotter": {"limit": 16},
    "learning_rate": 0.00001,
    "weights_decay": 0.00005,
    "layers": [784, 784],
    "data_paths": {"arial": os.path.join(root.common.test_dataset_root,
                                         "arial.ttf")}})
