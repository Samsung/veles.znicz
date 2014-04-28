#!/usr/bin/python3.3 -O
"""
Created on April 22, 2014

Convolitional relu channels config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

root.update = {"accumulator": {"n_bars": 30},
               "decision": {"fail_iterations": 1000,
                            "snapshot_prefix": "channels_relu_conv",
                            "use_dynamic_alpha": False,
                            "do_export_weights": True},
               "image_saver": {"out_dirs":
                               [os.path.join(root.common.cache_dir,
                                             "tmp/test"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/validation"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/train")]},
               "loader": {"cache_fnme":
                          os.path.join(root.common.cache_dir,
                                       "channels_relu_conv.pickle"),
                          "grayscale": False,
                          "minibatch_size": 40,
                          "n_threads": 32,
                          "channels_dir":
                          "/data/veles/VD/channels/russian_small/train",
                          "rect": (264, 129),
                          "validation_procent": 0.15},
               "weights_plotter": {"limit": 64},
               "channels_conv": {"export": False,
                                 "find_negative": 0,
                                 "global_alpha": 0.001,
                                 "global_lambda": 0.004,
                                 "layers":
                                 [{"type": "conv_relu", "n_kernels": 96,
                                   "kx": 11, "ky": 11, "sliding": (4, 4),
                                   "padding": (0, 0, 0, 0)},
                                  {"type": "max_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                                  {"type": "conv_relu", "n_kernels": 256,
                                   "kx": 5, "ky": 5, "sliding": (1, 1),
                                   "padding": (2, 2, 2, 2)},
                                  {"type": "max_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                                  {"type": "conv_relu", "n_kernels": 384,
                                   "kx": 3, "ky": 3, "sliding": (1, 1),
                                   "padding": (1, 1, 1, 1)},
                                  {"type": "conv_relu", "n_kernels": 384,
                                   "kx": 3, "ky": 3, "sliding": (1, 1),
                                   "padding": (1, 1, 1, 1)},
                                  {"type": "conv_relu", "n_kernels": 256,
                                   "kx": 3, "ky": 3, "sliding": (1, 1),
                                   "padding": (1, 1, 1, 1)},
                                  {"type": "max_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                                  {"type": "relu", "layers": 10}],
                                 "snapshot": ""}}
