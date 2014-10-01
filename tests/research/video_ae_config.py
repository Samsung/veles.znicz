#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Wine config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

root.video_ae.update({
    "decision": {"fail_iterations": 100},
    "snapshotter": {"prefix": "video_ae"},
    "loader": {"minibatch_size": 50},
    "weights_plotter": {"limit": 16},
    "learning_rate": 0.01,
    "weights_decay": 0.00005,
    "layers": [9, [90, 160]],
    "data_paths": os.path.join(root.common.test_dataset_root, "video_ae/img")})
