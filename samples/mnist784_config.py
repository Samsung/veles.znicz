#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters
root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "mnist_784"},
               "loader": {"minibatch_maxsize": 100},
               "weights_plotter": {"limit": 16},
               "mnist784": {"global_alpha": 0.001,
                            "global_lambda": 0.00005,
                            "layers": [784, 784],
                            "path_for_load_data":
                            os.path.join(root.common.test_dataset_root,
                                         "arial.ttf")}}
