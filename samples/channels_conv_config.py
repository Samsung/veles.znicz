#!/usr/bin/python3.3 -O
"""
Created on April 22, 2014

Convolitional channels config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

root.update = {"accumulator": {"n_bars": 30},
               "decision": {"fail_iterations": 1000,
                            "snapshot_prefix": "channels_conv",
                            "use_dynamic_alpha": False,
                            "do_export_weights": True},
               "image_saver": {"out_dirs":
                               [os.path.join(root.common.cache_dir,
                                             "tmp/test"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/validation"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/train")]},
               "loader": {"cache_fnme": os.path.join(root.common.cache_dir,
                                                     "channels_conv.pickle"),
                          "grayscale": False,
                          "minibatch_size": 81,
                          "n_threads": 32,
                          "channels_dir":
                          "/data/veles/VD/channels/russian_small/train",
                          "rect": (264, 129),
                          "validation_ratio": 0.15},
               "weights_plotter": {"limit": 64},
               "channels": {"export": False,
                                 "find_negative": 0,
                                 "global_alpha": 0.001,
                                 "global_lambda": 0.004,
                                 "layers":
                                 [{"type": "conv", "n_kernels": 32,
                                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                  {"type": "max_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                                  {"type": "conv", "n_kernels": 32,
                                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                  {"type": "avg_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                                  {"type": "conv", "n_kernels": 64,
                                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                  {"type": "avg_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                                  {"type": "softmax", "layers": 10}],
                                 "snapshot": ""}}
