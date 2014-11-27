#!/usr/bin/python3 -O
"""
Created on Nov 13, 2014

Configuration file for yale-faces (Self-constructing Model).
Model - fully connected neural network.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root


root.yalefaces.update({
    "decision": {"fail_iterations": 50, "max_epochs": 1000},
    "loss_function": "softmax",
    "snapshotter": {"prefix": "yalefaces"},
    "loader": {"minibatch_size": 40, "on_device": True,
               "validation_ratio": 0.15,
               "common_dir": root.common.test_dataset_root,
               "url":
               "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/"
               "CroppedYale.zip"},
    "layers": [{"type": "all2all_tanh", "learning_rate": 0.01,
                "weights_decay": 0.00005, "output_shape": 100},
               {"type": "softmax", "output_shape": 39, "learning_rate": 0.01,
                "weights_decay": 0.00005}]})

root.yalefaces.loader.data_dir = os.path.join(
    root.yalefaces.loader.common_dir, "CroppedYale")
