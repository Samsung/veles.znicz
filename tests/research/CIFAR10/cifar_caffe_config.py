#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for cifar (Self-constructing Model).
Model - convolutional neural network. Configuration parameters just like
in CAFFE.

Converged to 17.26% errors with some seed,
and below 18% in most cases.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")
root.common.precision_type = "float"
root.common.engine.backend = "cuda"

root.cifar.update({
    "decision": {"fail_iterations": 250, "max_epochs": 1000000000},
    "learning_rate_adjust": {"do": True},
    "snapshotter": {"prefix": "cifar_caffe", "interval": 1},
    "loss_function": "softmax",
    "add_plotters": False,
    "image_saver": {"do": False,
                    "out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 100,
               "norm": "mean",
               "add_sobel": False,
               "shuffle_limit": 2000000000,
               "on_device": True},
    "softmax": {"error_function_avr": True},
    "weights_plotter": {"limit": 64},
    "layers": [{"name": "conv1",
                "type": "conv", "n_kernels": 32, "kx": 5, "ky": 5,
                "padding": (2, 2, 2, 2), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.0001,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.001, "learning_rate_bias": 0.002,
                "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                "factor_ortho": 0.001, "gradient_moment": 0.9,
                "gradient_moment_bias": 0.9},

               {"name": "pool1", "type": "max_pooling",
                "kx": 3, "ky": 3, "sliding": (2, 2)},

               {"name": "relu1", "type": "activation_str"},

               {"name": "norm1", "type": "norm", "alpha": 0.00005,
                "beta": 0.75, "n": 3, "k": 1},

               {"name": "conv2", "type": "conv", "n_kernels": 32,
                "kx": 5, "ky": 5, "padding": (2, 2, 2, 2), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.001, "learning_rate_bias": 0.002,
                "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                "factor_ortho": 0.001, "gradient_moment": 0.9,
                "gradient_moment_bias": 0.9},

               {"name": "relu2", "type": "activation_str"},

               {"name": "pool2", "type": "avg_pooling",
                "kx": 3, "ky": 3, "sliding": (2, 2)},

               {"name": "norm2", "type": "norm",
                "alpha": 0.00005, "beta": 0.75, "n": 3, "k": 1},

               {"name": "conv3", "type": "conv", "n_kernels": 64,
                "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                "sliding": (1, 1), "weights_filling": "gaussian",
                "weights_stddev": 0.01, "bias_filling": "constant",
                "bias_stddev": 0, "learning_rate": 0.001,
                "learning_rate_bias": 0.001, "weights_decay": 0.0005,
                "weights_decay_bias": 0.0005, "factor_ortho": 0.001,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9},

               {"name": "relu3", "type": "activation_str"},

               {"name": "pool3", "type": "avg_pooling", "kx": 3, "ky": 3,
                "sliding": (2, 2)},

               {"name": "a2asm4", "type": "softmax", "output_shape": 10,
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.001, "learning_rate_bias": 0.002,
                "weights_decay": 1.0, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9}],
    "data_paths": {"train": train_dir, "validation": validation_dir}})
