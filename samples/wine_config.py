#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Wine config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root
import veles.random_generator as rnd


# optional parameters

root.common.update = {"plotters_disabled": True}

root.update = {"decision": {"fail_iterations": 200,
                            "snapshot_prefix": "wine"},
               "loader": {"minibatch_maxsize": 10,
                          "rnd": rnd.get(),
                          "view_group": "LOADER"},
               "wine": {"learning_rate": 0.1,
                        "weights_decay": 0.0,
                        "layers": [8, 3],
                        "data_paths":
                        os.path.join(root.common.veles_dir,
                                     "veles/znicz/samples/wine/wine.data")}}
