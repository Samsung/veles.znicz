# -*-coding: utf-8 -*-
"""
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


from veles.config import root
from veles.genetics import Tune


root.mnistr.update({
    "lr_adjuster": {"do": False},
    "decision": {"fail_iterations": 50,
                 "max_epochs": 1000000000},
    "loss_function": "softmax",
    "loader_name": "mnist_loader",
    "snapshotter": {"prefix": "mnist", "time_interval": 0, "compression": ""},
    "loader": {"minibatch_size": Tune(60, 1, 1000), "force_cpu": False,
               "normalization_type": "linear"},
    "weights_plotter": {"limit": 64},
    "layers": [{"type": "all2all_tanh",
                "->": {"output_sample_shape": Tune(100, 10, 500),
                       "weights_filling": "uniform",
                       "weights_stddev": Tune(0.05, 0.0001, 0.1),
                       "bias_filling": "uniform",
                       "bias_stddev": Tune(0.05, 0.0001, 0.1)},
                "<-": {"learning_rate": Tune(0.03, 0.0001, 0.9),
                       "weights_decay": Tune(0.0, 0.0, 0.9),
                       "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                       "weights_decay_bias": Tune(0.0, 0.0, 0.9),
                       "gradient_moment": Tune(0.0, 0.0, 0.95),
                       "gradient_moment_bias": Tune(0.0, 0.0, 0.95),
                       "factor_ortho": Tune(0.001, 0.0, 0.1)}},
               {"type": "softmax",
                "->": {"output_sample_shape": 10,
                       "weights_filling": "uniform",
                       "weights_stddev": Tune(0.05, 0.0001, 0.1),
                       "bias_filling": "uniform",
                       "bias_stddev": Tune(0.05, 0.0001, 0.1)},
                "<-": {"learning_rate": Tune(0.03, 0.0001, 0.9),
                       "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                       "weights_decay": Tune(0.0, 0.0, 0.95),
                       "weights_decay_bias": Tune(0.0, 0.0, 0.95),
                       "gradient_moment": Tune(0.0, 0.0, 0.95),
                       "gradient_moment_bias": Tune(0.0, 0.0, 0.95)}}]})
