#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for video_ae. Model - autoencoder.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters


root.video_ae.update({
    "decision": {"fail_iterations": 100, "max_epochs": 100000},
    "snapshotter": {"prefix": "video_ae"},
    "loader": {"minibatch_size": 50, "force_cpu": False,
               "train_paths":
               (os.path.join(root.common.test_dataset_root, "video_ae/img"),),
               "color_space": "GRAY",
               "background_color": (0x80,),
               "normalization_type": "linear"
               },
    "weights_plotter": {"limit": 16},
    "learning_rate": 0.01,
    "weights_decay": 0.00005,
    "layers": [9, [90, 160]]})
