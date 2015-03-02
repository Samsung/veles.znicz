#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for approximator.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

target_dir = [os.path.join(root.common.test_dataset_root,
                           "approximator/all_org_apertures.mat")]
train_dir = [os.path.join(root.common.test_dataset_root,
                          "approximator/all_dec_apertures.mat")]

root.approximator.update({
    "decision": {"fail_iterations": 1000, "max_epochs": 1000000000},
    "snapshotter": {"prefix": "approximator"},
    "loader": {"minibatch_size": 100, "train_paths": train_dir,
               "target_paths": target_dir,
               "normalization_type": "mean_disp",
               "target_normalization_type": "mean_disp"},
    "learning_rate": 0.0001,
    "weights_decay": 0.00005,
    "layers": [810, 9]})
