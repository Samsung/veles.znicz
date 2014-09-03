#!/usr/bin/python3 -O
# encoding: utf-8

"""
Created on Apr 18, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root


# optional parameters


root.update = {
    "decision": {"fail_iterations": 100,
                 "store_samples_mse": True},
    "snapshotter": {"prefix": "imagenet_caffe"},
    "loader": {"minibatch_size": 60},
    "imagenet_caffe": {"learning_rate": 0.00016,
                       "weights_decay": 0.0,
                       "layers":
                       [{"type": "conv_relu", "n_kernels": 96,
                         "kx": 11, "ky": 11, "padding": (0, 0, 0, 0),
                         "sliding": (4, 4),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},
                        {"type": "max_pooling",
                         "kx": 3, "ky": 3, "sliding": (2, 2)},
                        {"type": "norm", "alpha": 0.00005,
                         "beta": 0.75, "n": 3},

                        {"type": "conv_relu", "n_kernels": 256,
                         "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                         "sliding": (1, 1),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},
                        {"type": "max_pooling",
                         "kx": 3, "ky": 3, "sliding": (2, 2)},

                        {"type": "conv", "n_kernels": 384,
                         "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                         "sliding": (1, 1),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},

                        {"type": "conv", "n_kernels": 384,
                         "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                         "sliding": (1, 1),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},

                        {"type": "conv_relu", "n_kernels": 256,
                         "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                         "sliding": (1, 1),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},
                        {"type": "max_pooling",
                         "kx": 3, "ky": 3, "sliding": (2, 2)},

                        {"type": "all2all_relu", "output_shape": 4096,
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.005},

                        {"type": "dropout", "dropout_ratio": 0.5},

                        {"type": "all2all_relu", "output_shape": 4096,
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.005},

                        {"type": "dropout", "dropout_ratio": 0.5},

                        {"type": "softmax", "output_shape": 1000,
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01}]}}
