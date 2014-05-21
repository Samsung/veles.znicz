#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Kanji config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import six

from veles.config import root


# optional parameters

root.update = {
    "decision": {"fail_iterations": 1000,
                 "store_samples_mse": True},
    "loader": {"minibatch_maxsize": 5103,
               "validation_ratio": 0.15},
    "snapshotter": {"prefix": "kanji"},
    "weights_plotter": {"limit": 16},
    "kanji": {"learning_rate": 0.001,
              "weights_decay": 0.00005,
              "layers": [5103, 2889, 24 * 24],
              "data_paths":
              {"target":
               os.path.join(root.common.test_dataset_root,
                            ("kanji/target/targets.%d.pickle" %
                             (3 if six.PY3 else 2))),
               "train": os.path.join(root.common.test_dataset_root,
                                     "kanji/train")},
              "index_map": os.path.join(root.kanji.data_paths.train,
                                        "index_map.%d.pickle" %
                                        (3 if six.PY3 else 2))}}
