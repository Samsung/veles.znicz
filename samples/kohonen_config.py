#!/usr/bin/python3.3 -O
"""
Created on May 12, 2014

Example of Kohonen map demo on a sample dataset.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root

data_path = os.path.join(os.path.dirname(__file__), "kohonen")


root.update = {
    "forward": {"shape": (8, 8),
                "weights_stddev": 0.05,
                "weights_filling": "uniform"},
    "decision": {"snapshot_prefix": "kohonen",
                 "epochs": 160},
    "loader": {"minibatch_size": 10,
               "dataset_file": os.path.join(data_path, "kohonen.txt")},
    "train": {"gradient_decay": lambda t: 0.05 / (1.0 + t * 0.01),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.01)}}
