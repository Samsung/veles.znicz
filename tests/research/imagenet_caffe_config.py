#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on Apr 18, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root


# optional parameters


root.update = {"all2all": {"weights_stddev": 0.05},
               "decision": {"fail_iterations": 100,
                            "snapshot_prefix": "imagenet_caffe",
                            "store_samples_mse": True},
               "loader": {"minibatch_maxsize": 60},
               "imagenet_caffe": {"learning_rate": 0.01,
                                  "weights_decay": 0.0,
                                  "layers": [100, 10]}}
