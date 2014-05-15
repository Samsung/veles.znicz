#!/usr/bin/python3.3 -O
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

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "cifar"},
               "image_saver": {"out_dirs":
                               [os.path.join(root.common.cache_dir,
                                             "tmp/test"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/validation"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/train")]},
               "loader": {"minibatch_maxsize": 180},
               "accumulator": {"n_bars": 30},
               "weights_plotter": {"limit": 25},
               "cifar": {"learning_rate": 0.1,
                         "weights_decay": 0.00005,
                         "layers": [{"type": "all2all_tanh",
                                     "output_shape": 100},
                                    {"type": "softmax", "output_shape": 10}],
                         "data_paths": {"train": train_dir,
                                        "validation": validation_dir}}}
