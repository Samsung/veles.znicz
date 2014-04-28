#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Wine config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "video_ae"},
               "loader": {"minibatch_maxsize": 50},
               "weights_plotter": {"limit": 16},
               "video_ae": {"global_alpha": 0.0002,
                            "global_lambda": 0.00005,
                            "layers": [9, 14400],
                            "path_for_load_data":
                            os.path.join(root.common.test_dataset_root,
                                         "video/video_ae/img/*.png")}}
