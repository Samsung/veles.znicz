#!/usr/bin/python3 -O
"""
Created on Nov 20, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root

root.common.precision_type = "float"
root.common.precision_level = 0

root.imagenet.update({
    "decision": {"fail_iterations": 10000,
                 "max_epochs": 10},
    "loss_function": "softmax",
    "snapshotter": {"prefix": "cnna", "interval": 10},
    "loader": {"minibatch_size": 32, "on_device": False,
               "validation_ratio": 0.5, "shuffle_limit": 1,
               "sx": 227, "sy": 227},
    "layers": [{"type": "conv_str", "n_kernels": 64, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},
               {"type": "max_pooling", "kx": 2, "ky": 2,
                "sliding": (2, 2)},

               {"type": "conv_str", "n_kernels": 128, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},
               {"type": "max_pooling", "kx": 2, "ky": 2,
                "sliding": (2, 2)},

               {"type": "conv_str", "n_kernels": 256, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},

               {"type": "conv_str", "n_kernels": 256, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},
               {"type": "max_pooling", "kx": 2, "ky": 2,
                "sliding": (2, 2)},

               {"type": "conv_str", "n_kernels": 512, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},

               {"type": "conv_str", "n_kernels": 512, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},
               {"type": "max_pooling", "kx": 2, "ky": 2,
                "sliding": (2, 2)},

               {"type": "conv_str", "n_kernels": 512, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},

               {"type": "conv_str", "n_kernels": 512, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},
               {"type": "max_pooling", "kx": 2, "ky": 2,
                "sliding": (2, 2)},

               {"type": "all2all", "output_shape": 4096,
                "weights_filling": "gaussian", "weights_stddev": 0.005,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.01, "learning_rate_bias": 0.02},
               {"type": "activation_str"},
               {"type": "dropout", "dropout_ratio": 0.5},

               {"type": "all2all", "output_shape": 4096,
                "weights_filling": "gaussian", "weights_stddev": 0.005,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.001, "learning_rate_bias": 0.002},
               {"type": "activation_str"},
               {"type": "dropout", "dropout_ratio": 0.5},

               {"type": "softmax", "output_shape": 1000,
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": 0.001, "learning_rate_bias": 0.002}]})
