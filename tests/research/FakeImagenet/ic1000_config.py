#!/usr/bin/python3 -O
"""
Created on Nov 20, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root

root.common.precision_type = "float"
root.common.precision_level = 0

base_lr = 0.01
wd = 0.0005

root.imagenet.update({
    "loader_name": "fake_imagenet_loader",
    "decision": {"fail_iterations": 10000,
                 "max_epochs": 10},
    "snapshotter": {"prefix": "imagenet", "interval": 10},
    "add_plotters": True,
    "loss_function": "softmax",
    "loader": {"minibatch_size": 256, "shuffle_limit": 1, "sx": 227,
               "sy": 227},
    "weights_plotter": {"limit": 64},
    "layers": [{"type": "conv_str",
                "->": {"n_kernels": 96, "kx": 11, "ky": 11,
                       "padding": (0, 0, 0, 0), "sliding": (4, 4),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               {"type": "max_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},
               {"type": "norm", "n": 5, "alpha": 0.0001, "beta": 0.75},

               {"type": "zero_filter",
                "grouping": 2},
               {"type": "conv_str",
                "->": {"n_kernels": 256, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               {"type": "max_pooling", "->": {"kx": 3, "ky": 3,
                                              "sliding": (2, 2)}},
               {"type": "norm", "n": 5, "alpha": 0.0001, "beta": 0.75},

               {"type": "zero_filter", "grouping": 2},
               {"type": "conv_str",
                "->": {"n_kernels": 384, "kx": 3, "ky": 3,
                       "padding": (1, 1, 1, 1), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},

               {"type": "zero_filter", "grouping": 2},
               {"type": "conv_str",
                "->": {"n_kernels": 384, "kx": 3, "ky": 3,
                       "padding": (1, 1, 1, 1), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},

               {"type": "zero_filter", "grouping": 2},
               {"type": "conv_str",
                "->": {"n_kernels": 256, "kx": 3, "ky": 3,
                       "padding": (1, 1, 1, 1), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               {"type": "max_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"type": "all2all_relu",
                "->": {"output_sample_shape": 4096,
                       "weights_filling": "gaussian", "weights_stddev": 0.005,
                       "bias_filling": "constant", "bias_stddev": 1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               {"type": "dropout", "dropout_ratio": 0.5},

               {"type": "all2all_relu",
                "->": {"output_sample_shape": 4096,
                       "weights_filling": "gaussian", "weights_stddev": 0.005,
                       "bias_filling": "constant", "bias_stddev": 1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               {"type": "dropout", "dropout_ratio": 0.5},

               {"type": "softmax",
                "->": {"output_sample_shape": 1000,
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}]})
