#!/usr/bin/python3 -O
"""
Created on Mart 26, 2014

Configuration file for hands.
Model – fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root


# optional parameters
train_dir = [os.path.join(root.common.test_dataset_root,
                          "hands/Positive/Training"),
             os.path.join(root.common.test_dataset_root,
                          "hands/Negative/Training")]
validation_dir = [os.path.join(root.common.test_dataset_root,
                               "hands/Positive/Testing"),
                  os.path.join(root.common.test_dataset_root,
                               "hands/Negative/Testing")]

root.hands.update({
    "decision": {"fail_iterations": 100,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "hands"},
    "loader": {"minibatch_size": 60},
    "learning_rate": 0.0008,
    "weights_decay": 0.0,
    "layers": [30, 2],
    "data_paths": {"train": train_dir, "validation": validation_dir}})
