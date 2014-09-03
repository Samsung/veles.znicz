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

root.model = "tanh"

root.update = {
    "accumulator": {"bars": 30},
    "decision": {"fail_iterations": 1000,
                 "max_epochs": 10000,
                 "use_dynamic_alpha": False,
                 "do_export_weights": True},
    "snapshotter": {"prefix": "channels %s" % root.model},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir,
                                  "tmp %s/test" % root.model),
                     os.path.join(root.common.cache_dir,
                                  "tmp %s/validation" % root.model),
                     os.path.join(root.common.cache_dir,
                                  "tmp %s/train" % root.model)]},
    "loader": {"cache_file_name": os.path.join(root.common.cache_dir,
                                               "channels_%s.%d.pickle" %
                                               (root.model,
                                                sys.version_info[0])),
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
                 "learning_rate": 0.001,
                 "weights_decay": 0.00005,
                 "layers": [{"type": "all2all_tanh",
                             "output_shape": 54},
                            {"type": "softmax",
                             "output_shape": 11}],
                 "snapshot": ""}}
