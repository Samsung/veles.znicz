#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.cifar.update({
    "decision": {"fail_iterations": 1000},
    "learning_rate_adjust": {"do": False},
    "snapshotter": {"prefix": "cifar"},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 81},
    "accumulator": {"n_bars": 30},
    "weights_plotter": {"limit": 25},
    "layers": [{"type": "all2all", "output_shape": 486,
                "learning_rate": 0.0005, "weights_decay": 0.0},
               {"type": "activation_sincos"},
               {"type": "all2all", "output_shape": 486,
                "learning_rate": 0.0005, "weights_decay": 0.0},
               {"type": "activation_sincos"},
               {"type": "softmax", "output_shape": 10,
                "learning_rate": 0.0005, "weights_decay": 0.0}],
    "data_paths": {"train": train_dir, "validation": validation_dir}})
