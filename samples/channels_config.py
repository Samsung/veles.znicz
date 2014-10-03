#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Channels config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import sys

from veles.config import root


# optional parameters

root.channels.model = "tanh"

root.channels.update({
    "accumulator": {"bars": 30},
    "decision": {"fail_iterations": 1000,
                 "max_epochs": 10000},
    "snapshotter": {"prefix": "channels_%s" % root.channels.model},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir,
                                  "tmp_%s/test" % root.channels.model),
                     os.path.join(root.common.cache_dir,
                                  "tmp_%s/validation" % root.channels.model),
                     os.path.join(root.common.cache_dir,
                                  "tmp_%s/train" % root.channels.model)]},
    "loader": {"cache_file_name": os.path.join(root.common.cache_dir,
                                               "channels_%s.%d.pickle" %
                                               (root.channels.model,
                                                sys.version_info[0])),
               "grayscale": False,
               "minibatch_size": 81,
               "n_threads": 32,
               "channels_dir":
               "/data/veles/VD/channels/russian_small/train",
               "rect": (264, 129),
               "validation_ratio": 0.15},
    "weights_plotter": {"limit": 16},
    "export": False,
    "find_negative": 0,
    "learning_rate": 0.001,
    "weights_decay": 0.00005,
    "layers": [{"type": "all2all_tanh", "output_shape": 54},
               {"type": "softmax", "output_shape": 11}],
    "snapshot": ""})
