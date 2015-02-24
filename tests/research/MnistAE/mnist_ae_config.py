#!/usr/bin/python3 -O
"""
Created on Mar 20, 2013

Config for Model for digits recognition. Database - MNIST. Model - autoencoder.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root


root.mnist_ae.update({
    "all2all": {"weights_stddev": 0.05},
    "decision": {"fail_iterations": 20,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "mnist", "time_interval": 0, "compress": ""},
    "loader": {"minibatch_size": 100, "force_cpu": False,
               "normalization_type": "linear"},
    "learning_rate": 0.000001,
    "weights_decay": 0.00005,
    "gradient_moment": 0.00001,
    "weights_plotter": {"limit": 16},
    "pooling": {"kx": 3, "ky": 3, "sliding": (2, 2)},
    "include_bias": False,
    "unsafe_padding": True,
    "n_kernels": 5,
    "kx": 5,
    "ky": 5})
