#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters

root.update = {
    "learning_rate_adjust": {"do": True},
    "decision": {"fail_iterations": 100},
    "snapshotter": {"prefix": "mnist_caffe"},
    "loader": {"minibatch_size": 5},
    "weights_plotter": {"limit": 64},
    "mnist": {"learning_rate": 0.01, "gradient_moment": 0.9,
              "weights_decay": 0.0005,
              "layers":
              [{"type": "conv", "n_kernels": 20, "kx": 5, "ky": 5,
                "sliding": (1, 1), "learning_rate": 0.01,
                "learning_rate_bias": 0.02, "gradient_moment": 0.9,
                "gradient_moment_bias": 0,
                "weights_filling": "uniform",
                "bias_filling": "constant", "bias_stddev": 0,
                "weights_decay": 0.0005, "weights_decay_bias": 0},

               {"type": "max_pooling",
                "kx": 2, "ky": 2, "sliding": (2, 2)},

               {"type": "conv", "n_kernels": 50, "kx": 5, "ky": 5,
                "sliding": (1, 1), "learning_rate": 0.01,
                "learning_rate_bias": 0.02, "gradient_moment": 0.9,
                "gradient_moment_bias": 0,
                "weights_filling": "uniform",
                "bias_filling": "constant", "bias_stddev": 0,
                "weights_decay": 0.0005, "weights_decay_bias": 0.0},

               {"type": "max_pooling",
                "kx": 2, "ky": 2, "sliding": (2, 2)},

               {"type": "all2all_relu", "output_shape": 500,
                "learning_rate": 0.01, "learning_rate_bias": 0.02,
                "gradient_moment": 0.9, "gradient_moment_bias": 0,
                "weights_filling": "uniform",
                "bias_filling": "constant", "bias_stddev": 0,
                "weights_decay": 0.0005, "weights_decay_bias": 0.0},

               {"type": "softmax", "output_shape": 10,
                "learning_rate": 0.01, "learning_rate_bias": 0.02,
                "gradient_moment": 0.9, "gradient_moment_bias": 0,
                "weights_filling": "uniform",
                "bias_filling": "constant", "weights_decay": 0.0005,
                "weights_decay_bias": 0.0}]}}
