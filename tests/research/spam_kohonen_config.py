#!/usr/bin/python3.3 -O
"""
Created on May 12, 2014

Example of Kohonen map demo on a sample dataset.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root

spam_dir = os.path.join(os.path.dirname(__file__), "spam")

root.update = {
    "forward": {"shape": (8, 8),
                "weights_stddev": 0.05,
                "weights_filling": "uniform"},
    "decision": {"snapshot_prefix": "spam_kohonen",
                 "epochs": 160},
    "loader": {"minibatch_maxsize": 60,
               "file": os.path.join(spam_dir, "data.txt.xz"),
               "validation_ratio": 0},
    "train": {"gradient_decay": lambda t: 0.01 / (1.0 + t * 0.00001),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.00001)}}
