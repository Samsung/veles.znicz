#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Some convolutional config for Cifar.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

LR = 0.005
LRB = LR * 2
WD = 0.0005
WDB = WD
GM = 0.9
GMB = GM
WDSM = WD * 2
WDSMB = 0.0

root.cifar.update({
    "decision": {"fail_iterations": 250},
    "snapshotter": {"prefix": "cifar"},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 100,
               "shuffle_limit": 2000000000},
    "softmax": {"error_function_avr": True},
    "weights_plotter": {"limit": 64},
    "layers": [{"type": "conv", "n_kernels": 32,
                "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                "sliding": (1, 1),
                "weights_filling": "uniform",
                "bias_filling": "uniform",
                "learning_rate": LR, "learning_rate_bias": LRB,
                "weights_decay": WD, "weights_decay_bias": WDB,
                "gradient_moment": GM, "gradient_moment_bias": GMB},
               {"type": "norm", "alpha": 0.00005, "beta": 0.75,
                "n": 3, "k": 1},
               {"type": "activation_tanhlog"},
               {"type": "maxabs_pooling",
                "kx": 3, "ky": 3, "sliding": (2, 2)},

               {"type": "conv", "n_kernels": 32,
                "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                "sliding": (1, 1),
                "weights_filling": "uniform",
                "bias_filling": "uniform",
                "learning_rate": LR, "learning_rate_bias": LRB,
                "weights_decay": WD, "weights_decay_bias": WDB,
                "gradient_moment": GM, "gradient_moment_bias": GMB},
               {"type": "norm", "alpha": 0.00005, "beta": 0.75,
                "n": 3, "k": 1},
               {"type": "activation_tanhlog"},
               {"type": "avg_pooling",
                "kx": 3, "ky": 3, "sliding": (2, 2)},

               {"type": "conv", "n_kernels": 64,
                "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                "sliding": (1, 1),
                "weights_filling": "uniform",
                "bias_filling": "uniform",
                "learning_rate": LR, "learning_rate_bias": LRB,
                "weights_decay": WD, "weights_decay_bias": WDB,
                "gradient_moment": GM, "gradient_moment_bias": GMB},
               {"type": "activation_tanhlog"},
               {"type": "avg_pooling",
                "kx": 3, "ky": 3, "sliding": (2, 2)},

               {"type": "softmax", "output_shape": 10,
                "weights_filling": "uniform",
                "bias_filling": "uniform",
                "learning_rate": LR, "learning_rate_bias": LRB,
                "weights_decay": WDSM, "weights_decay_bias": WDSMB,
                "gradient_moment": GM, "gradient_moment_bias": GMB}],
    "data_paths": {"train": train_dir, "validation": validation_dir}})
