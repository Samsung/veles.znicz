#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for kanji.
Model – fully-connected Neural Network with MSE loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import sys

from veles.config import root


# optional parameters

train_path = os.path.join(root.common.test_dataset_root, "kanji/train")

root.kanji_standard.update({
    "decision": {"fail_iterations": 1000,
                 "max_epochs": 10000},
    "loss_function": "mse",
    "add_plotters": True,
    "loader": {"minibatch_size": 50,
               "validation_ratio": 0.15},
    "snapshotter": {"prefix": "kanji"},
    "weights_plotter": {"limit": 16},
    "layers": [{"type": "all2all_tanh", "learning_rate": 0.00001,
                "weights_decay": 0.00005, "output_shape": 250},
               {"type": "all2all_tanh", "learning_rate": 0.00001,
                "weights_decay": 0.00005, "output_shape": 250},
               {"type": "all2all_tanh", "output_shape": 24 * 24,
                "learning_rate": 0.00001,
                "weights_decay": 0.00005}],
    "data_paths": {"target": os.path.join(root.common.test_dataset_root,
                                          ("kanji/target/targets.%d.pickle" %
                                           (sys.version_info[0]))),
                   "train": train_path},
    "index_map": os.path.join(train_path, "index_map.%d.pickle" %
                              (sys.version_info[0]))})
