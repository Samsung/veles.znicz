#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for cifar (Self-constructing Model).
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.cifar.update({
    "loader_name": "cifar_loader",
    "decision": {"fail_iterations": 1000, "max_epochs": 1000000000},
    "lr_adjuster": {"do": False},
    "snapshotter": {"prefix": "cifar", "interval": 1},
    "loss_function": "softmax",
    "add_plotters": True,
    "image_saver": {"do": False,
                    "out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 81, "force_cpu": False,
               "normalization_type": "linear"},
    "accumulator": {"n_bars": 30},
    "weights_plotter": {"limit": 25},
    "similar_weights_plotter": {"form_threshold": 1.1, "peak_threshold": 0.5,
                                "magnitude_threshold": 0.65},
    "layers": [{"type": "all2all",
                "->": {"output_sample_shape": 486},
                "<-": {"learning_rate": 0.0005, "weights_decay": 0.0}},
               {"type": "activation_sincos"},
               {"type": "all2all",
                "->": {"output_sample_shape": 486},
                "<-": {"learning_rate": 0.0005, "weights_decay": 0.0}},
               {"type": "activation_sincos"},
               {"type": "softmax",
                "->": {"output_sample_shape": 10},
                "<-": {"learning_rate": 0.0005, "weights_decay": 0.0}}],
    "data_paths": {"train": train_dir, "validation": validation_dir}})
