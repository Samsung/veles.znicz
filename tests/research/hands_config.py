#!/usr/bin/python3.3 -O
"""
Created on Mart 26, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root


# optional parameters
train_dir = [os.path.join(root.common.test_dataset_root,
                          "hands/Positive/Training"),
             os.path.join(root.common.test_dataset_root,
                          "hands/Negative/Training")]
validation_dir = [os.path.join(root.common.test_dataset_root,
                               "hands/Positive/Testing"),
                  os.path.join(root.common.test_dataset_root,
                               "hands/Negative/Testing")]

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "hands"},
               "loader": {"minibatch_maxsize": 60},
               "hands": {"learning_rate": 0.05,
                         "weights_decay": 0.0,
                         "layers": [30, 2],
                         "data_paths": {"train": train_dir,
                                        "validation": validation_dir}}}
