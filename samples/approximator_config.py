#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Approximator config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters

target = ["/data/veles/approximator/all_org_appertures.mat"]
train = ["/data/veles/approximator/all_dec_appertures.mat"]

root.update = {"decision": {"fail_iterations": 1000,
                            "snapshot_prefix":  "approximator",
                            "store_samples_mse": True},
               "loader": {"minibatch_maxsize": 100},
               "approximator": {"global_alpha": 0.01,
                                "global_lambda": 0.00005,
                                "layers": [810, 9],
                                "path_for_load_data": {"target": target,
                                                       "train": train}}}
