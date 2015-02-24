#!/usr/bin/python3 -O
"""
Created on May 12, 2014

Example of Kohonen map demo on a sample dataset.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root

spam_dir = os.path.dirname(__file__)

root.spam_kohonen.update({
    "forward": {"shape": (8, 8)},
    "decision": {"epochs": 200},
    "loader": {"minibatch_size": 80,
               "force_cpu": True,
               "ids": True,
               "classes": False,
               "file": "/data/veles/VD/VDLogs/histogramConverter/data/hist",
               },
    "train": {"gradient_decay": lambda t: 0.002 / (1.0 + t * 0.00002),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.00002)},
    "exporter": {"file": "classified_fast4.txt"}})
