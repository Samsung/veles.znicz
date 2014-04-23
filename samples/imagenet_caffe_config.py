#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on Apr 18, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root, Config

root.all2all = Config()  # not necessary for execution (it will do it in real
root.decision = Config()  # time any way) but good for Eclipse editor
root.loader = Config()

# optional parameters


root.update = {"all2all": {"weights_magnitude": 0.05},
               "decision": {"fail_iterations": 100,
                            "snapshot_prefix": "mnist",
                            "store_samples_mse": True},
               "loader": {"minibatch_maxsize": 60},
               "imagenet_caffe": {"global_alpha": 0.01,
                         "global_lambda": 0.0,
                         "layers": [100, 10]}
               }
