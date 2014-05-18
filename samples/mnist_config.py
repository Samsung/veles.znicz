#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root

mnist_dir = os.path.join(os.path.dirname(__file__), "MNIST")

# optional parameters
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")


root.update = {"all2all": {"weights_stddev": 0.05},
               "decision": {"fail_iterations": 100,
                            "snapshot_prefix": "mnist"},
               "loader": {"minibatch_maxsize": 60},
               "mnist": {"learning_rate": 0.1,
                         "weights_decay": 0.0,
                         "layers": [100, 10],
                         "data_paths": {"test_images": test_image_dir,
                                        "test_label": test_label_dir,
                                        "train_images": train_image_dir,
                                        "train_label": train_label_dir}}}
