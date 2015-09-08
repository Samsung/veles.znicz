# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mart 21, 2014

Configuration file for Mnist with variation of parameters for genetic.
Model - fully-connected Neural Network with SoftMax loss function.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import os
from veles.config import root
from veles.genetics import Range


root.mnistr.update({
    "lr_adjuster": {"do": False},
    "decision": {"fail_iterations": 50,
                 "max_epochs": 1000000000},
    "loss_function": "softmax",
    "loader_name": "mnist_loader",
    "snapshotter": {"prefix": "mnist", "time_interval": 0, "compression": "gz",
                    # "odbc": "DRIVER={MySQL};SERVER=localhost;DATABASE=test;"
                    #         "UID=test;PWD=test",
                    },
    "loader": {"minibatch_size": Range(60, 1, 1000), "force_numpy": False,
               "normalization_type": "linear",
               "data_path": os.path.join(root.common.dirs.datasets, "MNIST")},
    "weights_plotter": {"limit": 64},
    "layers": [{"name": "fc_tanh1",
                "type": "all2all_tanh",
                "->": {"output_sample_shape": Range(100, 10, 500),
                       "weights_filling": "uniform",
                       "weights_stddev": Range(0.05, 0.0001, 0.1),
                       "bias_filling": "uniform",
                       "bias_stddev": Range(0.05, 0.0001, 0.1)},
                "<-": {"learning_rate": Range(0.03, 0.0001, 0.9),
                       "weights_decay": Range(0.0, 0.0, 0.9),
                       "learning_rate_bias": Range(0.03, 0.0001, 0.9),
                       "weights_decay_bias": Range(0.0, 0.0, 0.9),
                       "gradient_moment": Range(0.0, 0.0, 0.95),
                       "gradient_moment_bias": Range(0.0, 0.0, 0.95),
                       "factor_ortho": Range(0.001, 0.0, 0.1)}},
               {"name": "fc_softmax2",
                "type": "softmax",
                "->": {"output_sample_shape": 10,
                       "weights_filling": "uniform",
                       "weights_stddev": Range(0.05, 0.0001, 0.1),
                       "bias_filling": "uniform",
                       "bias_stddev": Range(0.05, 0.0001, 0.1)},
                "<-": {"learning_rate": Range(0.03, 0.0001, 0.9),
                       "learning_rate_bias": Range(0.03, 0.0001, 0.9),
                       "weights_decay": Range(0.0, 0.0, 0.95),
                       "weights_decay_bias": Range(0.0, 0.0, 0.95),
                       "gradient_moment": Range(0.0, 0.0, 0.95),
                       "gradient_moment_bias": Range(0.0, 0.0, 0.95)}}]})
