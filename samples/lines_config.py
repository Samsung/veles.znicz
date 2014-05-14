#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on May 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root


# optional parameters

root.update = {
               "decision": {"fail_iterations": 100,
                "snapshot_prefix": "lines"},
               "loader": {"minibatch_maxsize": 60},
               "weights_plotter": {"limit": 32},

               "lines": {"learning_rate": 0.01,
                         "weights_decay": 0.0,

                         "layers":
                         [
                          {
                           "type": "conv_relu", "n_kernels": 32,
                           "kx": 7, "ky": 7, "padding": (2, 2, 2, 2),
                           "weights_filling": "gaussian",
                           "weights_stddev": 0.0001
                           },


                          {
                           "type": "max_pooling",
                           "kx": 3, "ky": 3, "sliding": (2, 2)
                           },

                          {
                           "type": "conv_relu", "n_kernels": 16,
                           "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                           "weights_filling": "gaussian",
                           "weights_stddev": 0.001
                           },

                          {
                           "type": "all2all_relu", "output_shape": 100,
                           "weights_filling": "uniform",
                           "weights_stddev": 0.05
                           },

                          {"type": "softmax", "output_shape": 4}

                          ],

                         "snapshot": ""},

               "softmax": {"weights_filling": "uniform",
                           "weights_stddev": 0.05}}