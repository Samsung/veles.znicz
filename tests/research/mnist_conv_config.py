#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters

root.update = {"learning_rate_adjust": {"do": False},
               "decision": {"fail_iterations": 100},
               "snapshotter": {"prefix": "mnist_conv"},
               "loader": {"minibatch_size": 540},
               "weights_plotter": {"limit": 64},
               "mnist": {"learning_rate": 0.0000092,
                         "learning_rate_bias": 0,
                         "weights_decay": 0.00005,
                         "layers":
                         [{"type": "conv",
                           "n_kernels": 25,
                           "kx": 9, "ky": 9,
                           "learning_rate": 0.0000092,
                           "learning_rate_bias": 0,
                           "gradient_moment": 1.0e-30,
                           "gradient_moment_bias": 1.0e-30,
                           "weights_filling": "uniform",
                           #"weights_stddev": 0.0001,
                           "bias_filling": "uniform",
                           #"bias_stddev": 0.0001,
                           "weights_decay": 0.00005,
                           "weights_decay_bias": 0.0
                           },
                          #{"type": "avg_pooling",
                          # "kx": 2, "ky": 2, "sliding": (1, 1)},
                          {"type": "all2all_tanh",
                           "output_shape": 100,
                           "learning_rate": 0.0000092,
                           "learning_rate_bias": 0,
                           "gradient_moment": 1.0e-30,
                           "gradient_moment_bias": 1.0e-30,
                           "weights_filling": "uniform",
                           #"weights_stddev": 0.0001,
                           "bias_filling": "uniform",
                           #"bias_stddev": 0.0001,
                           "weights_decay": 0.00005,
                           "weights_decay_bias": 0.0},
                          {"type": "softmax",
                           "output_shape": 10,
                           "learning_rate": 0.0000092,
                           "learning_rate_bias": 0,
                           "gradient_moment": 1.0e-30,
                           "gradient_moment_bias": 1.0e-30,
                           "weights_filling": "uniform",
                           #"weights_stddev": 0.0001,
                           "bias_filling": "uniform",
                           #"bias_stddev": 0.0001,
                           "weights_decay": 0.00005,
                           "weights_decay_bias": 0.0}]}}
