#!/usr/bin/python3 -O
"""
Created on May 12, 2014

Example of Kohonen map demo on a sample dataset.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root

spam_dir = os.path.join(os.path.dirname(__file__), "spam")

root.update = {
    "forward": {"shape": (8, 8)},
    "decision": {"epochs": 400},
    "loader": {"minibatch_size": 100,
               # "file": os.path.join(spam_dir, "data.txt")},
               "ids": True,
               "classes": False,
               "file": "/data/veles/VDLogs/histogramConverter/data/hist"},
    "train": {"gradient_decay": lambda t: 0.001 / (1.0 + t * 0.000001),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.000001)},
    "exporter": {"file": "classified.txt"}}
