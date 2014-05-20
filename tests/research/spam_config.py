#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root


spam_dir = os.path.join(os.path.dirname(__file__), "spam")

root.update = {
    "decision": {"fail_iterations": 100,
                 "store_samples_mse": True},
    "snapshotter": {"prefix": "spam"},
    "loader": {"minibatch_maxsize": 60,
               "file": os.path.join(spam_dir, "data.txt.xz"),
               "validation_ratio": 0.15},
    "spam": {"learning_rate": 0.0001,
             "weights_decay": 0.00005,
             "gradient_moment": 0.9,
             "gradient_moment_bias": 0.9,
             "layers": [100, 2]}}
