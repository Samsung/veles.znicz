#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for Mnist. Model - fully-connected Neural Network with
SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root

mnist_dir = mnist_dir = "veles/znicz/samples/MNIST"

# optional parameters
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")


root.mnist.update({
    "all2all": {"weights_stddev": 0.05},
    "decision": {"fail_iterations": 300,
                 "snapshot_prefix": "mnist"},
    "loader": {"minibatch_size": 88, "force_cpu": False,
               "normalization_type": "linear"},
    "learning_rate": 0.028557478339518444,
    "weights_decay": 0.00012315096341168246,
    "factor_ortho": 0.001,
    "layers": [364, 10],
    "data_paths": {"test_images": test_image_dir,
                   "test_label": test_label_dir,
                   "train_images": train_image_dir,
                   "train_label": train_label_dir}})
