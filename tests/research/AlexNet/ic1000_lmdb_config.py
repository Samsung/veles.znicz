#!/usr/bin/python3 -O
"""
Created on Nov 20, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import os

from veles.config import root

root.common.precision_type = "float"
root.common.precision_level = 0

base_lr = 0.01
wd = 0.0005

data_path = "/data/veles/datasets/imagenet/temp/LMDB_old"

root.imagenet.update({
    "decision": {"fail_iterations": 10000,
                 "max_epochs": 10},
    "snapshotter": {"prefix": "imagenet", "interval": 10},
    "add_plotters": True,
    "loss_function": "softmax",
    "loader_name": "lmdb",
    "loader": {"minibatch_size": 256, "on_device": False,
               "shuffle_limit": 1, "crop": (227, 227), "mirror": True,
               "color_space": "HSV", "normalization_type": "pointwise",
               "train_path": os.path.join(data_path, "ilsvrc12_train_lmdb"),
               "validation_path": os.path.join(data_path, "ilsvrc12_val_lmdb"),
               "mean_rdisp_path": "data_mean_rdisp_path.npz"
               },

    "weights_plotter": {"limit": 64},
    "layers": [{"type": "conv_str", "n_kernels": 96, "kx": 11, "ky": 11,
                "padding": (0, 0, 0, 0), "sliding": (4, 4),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": base_lr, "learning_rate_bias": base_lr * 2,
                "weights_decay": wd, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
               {"type": "max_pooling", "kx": 3, "ky": 3,
                "sliding": (2, 2)},
               {"type": "norm", "n": 5, "alpha": 0.0001, "beta": 0.75},

               {"type": "conv_str", "n_kernels": 256, "kx": 5, "ky": 5,
                "padding": (2, 2, 2, 2), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 1,
                "learning_rate": base_lr, "learning_rate_bias": base_lr * 2,
                "weights_decay": wd, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
               {"type": "max_pooling", "kx": 3, "ky": 3,
                "sliding": (2, 2)},
               {"type": "norm", "n": 5, "alpha": 0.0001, "beta": 0.75},

               {"type": "conv_str", "n_kernels": 384, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": base_lr, "learning_rate_bias": base_lr * 2,
                "weights_decay": wd, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9},

               {"type": "conv_str", "n_kernels": 384, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 1,
                "learning_rate": base_lr, "learning_rate_bias": base_lr * 2,
                "weights_decay": wd, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9},

               {"type": "conv_str", "n_kernels": 256, "kx": 3, "ky": 3,
                "padding": (1, 1, 1, 1), "sliding": (1, 1),
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 1,
                "learning_rate": base_lr, "learning_rate_bias": base_lr * 2,
                "weights_decay": wd, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
               {"type": "max_pooling", "kx": 3, "ky": 3,
                "sliding": (2, 2)},

               {"type": "all2all_relu", "output_shape": 4096,
                "weights_filling": "gaussian", "weights_stddev": 0.005,
                "bias_filling": "constant", "bias_stddev": 1,
                "learning_rate": base_lr, "learning_rate_bias": base_lr * 2,
                "weights_decay": wd, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
               {"type": "dropout", "dropout_ratio": 0.5},

               {"type": "all2all_relu", "output_shape": 4096,
                "weights_filling": "gaussian", "weights_stddev": 0.005,
                "bias_filling": "constant", "bias_stddev": 1,
                "learning_rate": base_lr, "learning_rate_bias": base_lr * 2,
                "weights_decay": wd, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
               {"type": "dropout", "dropout_ratio": 0.5},

               {"type": "softmax", "output_shape": 1000,
                "weights_filling": "gaussian", "weights_stddev": 0.01,
                "bias_filling": "constant", "bias_stddev": 0,
                "learning_rate": base_lr, "learning_rate_bias": base_lr * 2,
                "weights_decay": wd, "weights_decay_bias": 0,
                "gradient_moment": 0.9, "gradient_moment_bias": 0.9}]})
