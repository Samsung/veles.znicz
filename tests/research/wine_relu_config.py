#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Wine config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

root.common.update = {"plotters_disabled": True}

root.update = {"decision": {"fail_iterations": 250,
                            "snapshot_prefix": "wine_relu"},
               "loader": {"minibatch_maxsize": 1000000},
               "wine_relu": {"learning_rate": 0.75,
                             "weights_decay": 0.0,
                             "layers": [10, 3],
                             "data_paths":
                             os.path.join(root.common.veles_dir,
                                          "veles/znicz/samples/wine/wine.data")
                             }}
