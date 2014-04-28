#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root

mnist_dir = "/data/veles/Veles2/veles/znicz/samples/MNIST"

# optional parameters

test_image_dir = os.path.join(
    root.common.veles_dir, "veles/znicz/samples/MNIST/t10k-images.idx3-ubyte")
test_label_dir = os.path.join(
    root.common.veles_dir, "veles/znicz/samples/MNIST/t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(
    root.common.veles_dir, "veles/znicz/samples/MNIST/train-images.idx3-ubyte")
train_label_dir = os.path.join(
    root.common.veles_dir, "veles/znicz/samples/MNIST/train-labels.idx1-ubyte")

root.update = {"all2all": {"weights_magnitude": 0.05},
               "decision": {"fail_iterations": 150,
                            "snapshot_prefix": "mnist_relu"},
               "loader": {"minibatch_maxsize": 60},
               "mnist_relu": {"global_alpha": 0.01,
                              "global_lambda": 0.0,
                              "layers": [100, 10],
                              "path_for_load_data": {"test_images":
                                                     test_image_dir,
                                                     "test_label":
                                                     test_label_dir,
                                                     "train_images":
                                                     train_image_dir,
                                                     "train_label":
                                                     train_label_dir}}}
