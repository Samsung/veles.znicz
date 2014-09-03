#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Approximator config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

target = [os.path.join(root.common.test_dataset_root,
                       "approximator/all_org_appertures.mat")]
train = [os.path.join(root.common.test_dataset_root,
                      "approximator/all_dec_appertures.mat")]

root.update = {"decision": {"fail_iterations": 1000,
                            "store_samples_mse": True},
               "snapshotter": {"prefix": "approximator"},
               "loader": {"minibatch_size": 100},
               "approximator": {"learning_rate": 0.0001,
                                "weights_decay": 0.00005,
                                "layers": [810, 9],
                                "data_paths": {"target": target,
                                               "train": train}}}
