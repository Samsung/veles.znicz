#!/usr/bin/python3 -O
"""
Created on Mart 26, 2014

Configuration file for hands.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root


# optional parameters
train_dir = [os.path.join(root.common.test_dataset_root, "hands/Training")]
validation_dir = [os.path.join(root.common.test_dataset_root, "hands/Testing")]


root.hands.update({
    "decision": {"fail_iterations": 100, "max_epochs": 10000},
    "loss_function": "softmax",
    "image_saver": {"do": True,
                    "out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader_name": "hands_loader",
    "snapshotter": {"prefix": "hands", "interval": 1, "time_interval": 0},
    "loader": {"minibatch_size": 40, "train_paths": train_dir,
               "force_cpu": False, "color_space": "GRAY",
               "background_color": (0,),
               "normalization_type": "linear",
               "validation_paths": validation_dir},
    "layers": [{"type": "all2all_tanh", "learning_rate": 0.008,
                "weights_decay": 0.0, "output_sample_shape": 30},
               {"type": "softmax", "output_sample_shape": 2,
                "learning_rate": 0.008, "weights_decay": 0.0}]})
