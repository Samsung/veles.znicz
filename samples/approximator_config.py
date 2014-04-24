#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Approximator config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root, Config

root.decision = Config()  # not necessary for execution (it will do it in real
root.loader = Config()  # time any way) but good for Eclipse editor

# optional parameters

target = ["/data/veles/approximator/all_org_appertures.mat"]
train = ["/data/veles/approximator/all_dec_appertures.mat"]

root.update = {"decision": {"fail_iterations": 1000,
                            "snapshot_prefix":  "approximator",
                            "store_samples_mse": True},
               "approximator": {"global_alpha": 0.01,
                                "global_lambda": 0.00005,
                                "layers": [810, 9],
                                "path_for_load_data": {"target": target,
                                                       "train": train}}}
