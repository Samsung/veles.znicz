# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Mar 20, 2013

Config for Model for digits recognition. Database - MNIST. Model - autoencoder.

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


root.mnist_ae.update({
    "all2all": {"weights_stddev": 0.05},
    "decision": {"fail_iterations": 20,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "mnist", "time_interval": 0, "compress": ""},
    "loader": {"minibatch_size": 100, "force_cpu": False,
               "normalization_type": "linear"},
    "learning_rate": 0.000001,
    "weights_decay": 0.00005,
    "gradient_moment": 0.00001,
    "weights_plotter": {"limit": 16},
    "pooling": {"kx": 3, "ky": 3, "sliding": (2, 2)},
    "include_bias": False,
    "unsafe_padding": True,
    "n_kernels": 5,
    "kx": 5,
    "ky": 5})
