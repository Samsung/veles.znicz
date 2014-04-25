#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Wine config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root, Config


root.decision = Config()  # not necessary for execution (it will do it in real
# time any way) but good for Eclipse editor

# optional parameters

root.common.update = {"plotters_disabled": True}

root.update = {"decision": {"fail_iterations": 250,
                            "snapshot_prefix": "wine_relu"},
               "loader": {"minibatch_maxsize": 1000000},
               "wine_relu": {"global_alpha": 0.75,
                             "global_lambda": 0.0,
                             "layers": [10, 3],
                             "path_for_load_data":
                             os.path.join(root.common.veles_dir,
                                          "veles/znicz/samples/wine/wine.data")
                             }}
