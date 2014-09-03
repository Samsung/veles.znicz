#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters
root.update = {"decision": {"fail_iterations": 100},
               "snapshotter": {"prefix": "mnist_784"},
               "loader": {"minibatch_size": 100},
               "weights_plotter": {"limit": 16},
               "mnist784": {"learning_rate": 0.00001,
                            "weights_decay": 0.00005,
                            "layers": [784, 784],
                            "data_paths":
                            os.path.join(root.common.test_dataset_root,
                                         "arial.ttf")}}
