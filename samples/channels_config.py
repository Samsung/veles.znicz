#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Channels config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

root.model = "tanh"

root.update = {"accumulator": {"n_bars": 30},
               "decision": {"fail_iterations": 1000,
                            "snapshot_prefix": "channels %s" % root.model,
                            "use_dynamic_alpha": False,
                            "do_export_weights": True},
               "conv":  {"weights_filling": "uniform"},
               "conv_relu":  {"weights_filling": "uniform"},
               "image_saver": {"out_dirs":
                               [os.path.join(root.common.cache_dir,
                                             "tmp %s/test" % root.model),
                                os.path.join(root.common.cache_dir,
                                             "tmp %s/validation" %
                                             root.model),
                                os.path.join(root.common.cache_dir,
                                             "tmp %s/train" % root.model)]},
               "loader": {"cache_fnme": os.path.join(root.common.cache_dir,
                                                     "channels_%s.pickle"
                                                     % root.model),
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
                            "layers": [{"type": "tanh", "layers": 54},
                                       {"type": "softmax", "layers": 11}],
                            "snapshot": ""}}
