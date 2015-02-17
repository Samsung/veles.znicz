#!/usr/bin/python3 -O
"""
Created on Nov 13, 2014

Configuration file for yale-faces (Self-constructing Model).
Model - fully connected neural network.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
from veles.config import root


root.yalefaces.update({
    "decision": {"fail_iterations": 50, "max_epochs": 1000},
    "loss_function": "softmax",
    "loader_name": "full_batch_auto_label_file_image",
    "snapshotter": {"prefix": "yalefaces", "interval": 1, "time_interval": 60},
    "loader": {"minibatch_size": 50, "on_device": True,
               "validation_ratio": 0.15,
               "filename_types": ["x-portable-graymap"],
               "ignored_files": [".*Ambient.*"],
               "shuffle_limit": numpy.iinfo(numpy.uint32).max,
               "add_sobel": False,
               "mirror": False,
               "color_space": "GRAY",
               "background_color": (0,),
               "normalization_type": "mean_disp",
               "train_paths":
               [os.path.join(root.common.test_dataset_root, "CroppedYale")]},
    "layers": [{"type": "all2all_tanh", "learning_rate": 0.01,
                "weights_decay": 0.00005, "output_sample_shape": 100},
               {"type": "softmax", "output_sample_shape": 39,
                "learning_rate": 0.01, "weights_decay": 0.00005}]})
