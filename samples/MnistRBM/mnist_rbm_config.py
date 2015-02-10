#!/usr/bin/python3 -O
"""
Created on Nov 20, 20134

Model created for digits recognition. Database - MNIST.
Model - RBM Neural Network.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


root.mnist_rbm.update({
    "all2all": {"weights_stddev": 0.05, "output_sample_shape": 1000},
    "decision": {"fail_iterations": 100,
                 "max_epochs": 100},
    "snapshotter": {"prefix": "mnist_rbm"},
    "loader": {"minibatch_size": 128, "on_device": True,
               "data_path":
               os.path.join(os.path.dirname(__file__), "..", "..",
                            "tests/unit/data/rbm/test_rbm.mat")},
    "learning_rate": 0.03,
    "weights_decay": 0.0005,
    "factor_ortho": 0.0})
