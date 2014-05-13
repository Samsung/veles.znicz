#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.update = {"decision": {"fail_iterations": 1000,
                            "snapshot_prefix": "cifar_caffe",
                            "do_export_weights": True},
               "image_saver": {"out_dirs":
                               [os.path.join(root.common.cache_dir,
                                             "tmp/test"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/validation"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/train")]},
               "loader": {"minibatch_maxsize": 100},
               "weights_plotter": {"limit": 64},
               "cifar_caffe": {"learning_rate": 0.001,
                               "weights_decay": 0.004,
                               "layers":
                               [{"type": "conv_relu", "n_kernels": 32,
                                 "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                {"type": "max_pooling",
                                 "kx": 3, "ky": 3, "sliding": (2, 2)},
                                {"type": "conv_relu", "n_kernels": 32,
                                 "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                {"type": "avg_pooling",
                                 "kx": 3, "ky": 3, "sliding": (2, 2)},
                                {"type": "conv_relu", "n_kernels": 64,
                                 "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                {"type": "avg_pooling",
                                 "kx": 3, "ky": 3, "sliding": (2, 2)}, 10],
                               "path_for_load_data": {"train": train_dir,
                                                      "validation":
                                                      validation_dir}}}
