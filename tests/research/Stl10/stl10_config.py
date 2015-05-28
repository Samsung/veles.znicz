"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 12, 2015

Configuration file for recognition of objects with STL-10 database.

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

root.stl.publisher.backends = {"confluence": {
    "server": "http://confluence.rnd.samsung.ru",
    "username": "al-jenkins", "password": "jenkins",
    "space": "VEL", "parent": "Veles"}}

root.stl.update({
    "loader_name": "full_batch_stl_10",
    "loss_function": "softmax",
    "downloader": {
        "url": "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
        "directory": root.common.datasets_root,
        "files": ["stl10_binary"]},
    "decision": {"fail_iterations": 20, "max_epochs": 100},
    "loader": {"directory":
               os.path.join(root.common.datasets_root, "stl10_binary"),
               "minibatch_size": 50,
               "scale": (32, 32),
               "normalization_type": "internal_mean"},
    "weights_plotter": {"limit": 256, "split_channels": False},
    "layers": [{"name": "conv1",
                "type": "conv",
                "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.0001,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                       "factor_ortho": 0.001, "gradient_moment": 0.9,
                       "gradient_moment_bias": 0.9},
                },
               {"name": "pool1", "type": "max_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"name": "relu1", "type": "activation_str"},

               {"name": "norm1", "type": "norm", "alpha": 0.00005,
                "beta": 0.75, "n": 3, "k": 1},

               {"name": "conv2", "type": "conv",
                "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                       "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                       "factor_ortho": 0.001, "gradient_moment": 0.9,
                       "gradient_moment_bias": 0.9}
                },
               {"name": "relu2", "type": "activation_str"},

               {"name": "pool2", "type": "avg_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"name": "norm2", "type": "norm",
                "alpha": 0.00005, "beta": 0.75, "n": 3, "k": 1},

               {"name": "conv3", "type": "conv",
                "->": {"n_kernels": 64, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2), "bias_stddev": 0,
                       "sliding": (1, 1), "weights_filling": "gaussian",
                       "weights_stddev": 0.01, "bias_filling": "constant"},
                "<-": {"learning_rate": 0.001,
                       "learning_rate_bias": 0.001, "weights_decay": 0.0005,
                       "weights_decay_bias": 0.0005, "factor_ortho": 0.001,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
                },
               {"name": "relu3", "type": "activation_str"},

               {"name": "pool3", "type": "avg_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"name": "a2asm4", "type": "softmax",
                "->": {"output_sample_shape": 10,
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                       "weights_decay": 1.0, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}]})
