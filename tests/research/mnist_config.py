#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for Mnist with variation of parameters for genetic.
Model – fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import os

from veles.config import root
from veles.genetics import Tune


mnist_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "samples/MNIST")
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")

root.mnistr.update({
    "learning_rate_adjust": {"do": False},
    "decision": {"fail_iterations": 100,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "mnist", "time_interval": 0, "compress": ""},
    "loader": {"minibatch_size": Tune(60, 1, 1000), "on_device": True},
    "weights_plotter": {"limit": 64},
    "layers": [{"type": "all2all_tanh", "output_shape": Tune(100, 10, 500),
                "learning_rate": Tune(0.03, 0.0001, 0.9),
                "weights_decay": Tune(0.0, 0.0, 0.9),
                "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                "weights_decay_bias": Tune(0.0, 0.0, 0.9),
                "gradient_moment": Tune(0.0, 0.0, 0.95),
                "gradient_moment_bias": Tune(0.0, 0.0, 0.95),
                "factor_ortho": Tune(0.001, 0.0, 0.1),
                "weights_filling": "uniform",
                "weights_stddev": Tune(0.05, 0.0001, 0.1),
                "bias_filling": "uniform",
                "bias_stddev": Tune(0.05, 0.0001, 0.1)},
               {"type": "softmax", "output_shape": 10,
                "learning_rate": Tune(0.03, 0.0001, 0.9),
                "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                "weights_decay": Tune(0.0, 0.0, 0.95),
                "weights_decay_bias": Tune(0.0, 0.0, 0.95),
                "gradient_moment": Tune(0.0, 0.0, 0.95),
                "gradient_moment_bias": Tune(0.0, 0.0, 0.95),
                "weights_filling": "uniform",
                "weights_stddev": Tune(0.05, 0.0001, 0.1),
                "bias_filling": "uniform",
                "bias_stddev": Tune(0.05, 0.0001, 0.1)}],
    "data_paths": {"test_images":  test_image_dir,
                   "test_label": test_label_dir,
                   "train_images": train_image_dir,
                   "train_label": train_label_dir}})
