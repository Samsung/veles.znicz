#!/usr/bin/python3 -O
"""
Created on May 12, 2014

Config for Kohonen map demo on a simple two dimension dataset.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root, get


data_path = os.path.abspath(get(
    root.kohonen.loader.base, os.path.dirname(__file__)))

root.kohonen.update({
    "forward": {"shape": (8, 8),
                "weights_stddev": 0.05,
                "weights_filling": "uniform"},
    "decision": {"snapshot_prefix": "kohonen",
                 "epochs": 200},
    "loader": {"minibatch_size": 10,
               "dataset_file": os.path.join(data_path, "kohonen.txt.gz"),
               "force_cpu": True},
    "train": {"gradient_decay": lambda t: 0.05 / (1.0 + t * 0.005),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.005)}})
