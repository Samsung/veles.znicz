#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on May 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root


# optional parameters


root.update = {"all2all": {"weights_magnitude": 0.05},
               "decision": {"fail_iterations": 100,
                            "snapshot_prefix": "lines",
                            "store_samples_mse": True},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 32},
               "lines": {"global_alpha": 0.01,
                         "global_lambda": 0.0,
                         "layers": [100, 10]}}
