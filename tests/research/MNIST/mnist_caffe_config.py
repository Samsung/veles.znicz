#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for Mnist. Configuration parameters just like
in CAFFE. Model - convolutional neural network.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters

root.mnistr.update({
    "loss_function": "softmax",
    "loader_name": "mnist_loader",
    "learning_rate_adjust": {"do": True},
    "decision": {"fail_iterations": 100, "max_epochs": 10000},
    "snapshotter": {"prefix": "mnist_caffe",
                    "time_interval": 0, "compress": ""},
    "loader": {"minibatch_size": 5, "force_cpu": False,
               "normalization_type": "linear"},
    "weights_plotter": {"limit": 64},
    "layers": [{"type": "conv",
                "->": {"n_kernels": 20, "kx": 5, "ky": 5,
                       "sliding": (1, 1), "weights_filling": "uniform",
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.01, "learning_rate_bias": 0.02,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0,
                       "weights_decay": 0.0005, "weights_decay_bias": 0}},

               {"type": "max_pooling",
                "->": {"kx": 2, "ky": 2, "sliding": (2, 2)}},

               {"type": "conv",
                "->": {"n_kernels": 50, "kx": 5, "ky": 5,
                       "sliding": (1, 1), "weights_filling": "uniform",
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.01, "learning_rate_bias": 0.02,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0}},

               {"type": "max_pooling",
                "->": {"kx": 2, "ky": 2, "sliding": (2, 2)}},

               {"type": "all2all_relu",
                "->": {"output_sample_shape": 500,
                       "weights_filling": "uniform",
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.01, "learning_rate_bias": 0.02,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0}},

               {"type": "softmax",
                "->": {"output_sample_shape": 10, "weights_filling": "uniform",
                       "bias_filling": "constant"},
                "<-": {"learning_rate": 0.01, "learning_rate_bias": 0.02,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0}}]})
