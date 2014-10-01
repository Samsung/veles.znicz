#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Kanji config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import sys

from veles.config import root


# optional parameters

train_path = os.path.join(root.common.test_dataset_root, "kanji/train")

root.kanji.update({
    "decision": {"fail_iterations": 1000,
                 "store_samples_mse": True},
    "loader": {"minibatch_size": 50,
               "validation_ratio": 0.15},
    "snapshotter": {"prefix": "kanji"},
    "weights_plotter": {"limit": 16},
    "learning_rate": 0.00001,
    "weights_decay": 0.00005,
    "layers": [250, 250, 24 * 24],
    "data_paths": {"target": os.path.join(root.common.test_dataset_root,
                                          ("kanji/target/targets.%d.pickle" %
                                           (sys.version_info[0]))),
                   "train": train_path},
    "index_map": os.path.join(train_path, "index_map.%d.pickle" %
                              (sys.version_info[0]))})
