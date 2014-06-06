#!/usr/bin/python3.3 -O
"""
Created on April 22, 2014

Convolitional relu channels config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

root.model = "relu_conv"

root.update = {"accumulator": {"bars": 30},
               "decision": {"fail_iterations": 1000,
                            "max_epochs": 10000,
                            "use_dynamic_alpha": False,
                            "do_export_weights": True},
               "snapshotter": {"prefix": "channels %s" % root.model},
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
                          "minibatch_size": 40,
                          "n_threads": 32,
                          "channels_dir":
                          "/data/veles/VD/channels/russian_small/train",
                          "rect": (264, 129),
                          "validation_procent": 0.15},
               "weights_plotter": {"limit": 64},
               "channels": {"export": False,
                            "find_negative": 0,
                            "learning_rate": 0.001,
                            "weights_decay": 0.004,
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
                             {"type": "softmax", "output_shape": 11}],
                            "snapshot": ""}}
