# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Mart 21, 2014

Configuration file for Mnist. Model - fully-connected Neural Network with
SoftMax loss function.

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

mnist_dir = "veles/znicz/samples/MNIST"

# optional parameters
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")


root.mnist.update({
    "all2all": {"weights_stddev": 0.05},
    "decision": {"fail_iterations": 300,
                 "snapshot_prefix": "mnist"},
    "loader": {"minibatch_size": 88, "force_numpy": False,
               "normalization_type": "linear"},
    "learning_rate": 0.028557478339518444,
    "weights_decay": 0.00012315096341168246,
    "factor_ortho": 0.001,
    "layers": [364, 10],
    "data_paths": {"test_images": test_image_dir,
                   "test_label": test_label_dir,
                   "train_images": train_image_dir,
                   "train_label": train_label_dir}})
