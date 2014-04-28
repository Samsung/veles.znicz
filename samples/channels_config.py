#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Channels config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root, Config

root.decision = Config()  # not necessary for execution (it will do it in real
root.loader = Config()  # time any way) but good for Eclipse editor

# optional parameters

root.update = {"decision": {"fail_iterations": 1000,
                            "snapshot_prefix": "channels_54_10",
                            "use_dynamic_alpha": False},
               "loader": {"cache_fnme": os.path.join(root.common.cache_dir,
                                                     "channels.pickle"),
                          "grayscale": False,
                          "minibatch_size": 81,
                          "n_threads": 32,
                          "channels_dir":
                          "/data/veles/VD/channels/russian_small/train",
                          "rect": (264, 129),
                          "validation_ratio": 0.15},
               "weights_plotter": {"limit": 16},
               "channels": {"export": False,
                            "find_negative": 0,
                            "global_alpha": 0.01,
                            "global_lambda": 0.00005,
                            "layers": [54, 10],
                            "snapshot": ""}}
