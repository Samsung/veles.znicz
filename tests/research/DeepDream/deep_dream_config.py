"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on August 27, 2015

Configuration file for DeepDream workflow

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

root.common.engine.precision_type = "float"

base_path = os.path.join(root.common.dirs.datasets, "deep_dream")
base_path_to_model = os.path.join(
    root.common.dirs.datasets, "AlexNet/imagenet")

root.deep_dream.update({
    "path_to_input": os.path.join(base_path, "input"),
    "path_to_output": os.path.join(base_path, "output"),
    "path_to_temp_output": os.path.join(base_path, "visualize"),
    "snapshot_name":
    "imagenet_validation_54.50_train_48.96.4.pickle.gz",
    "matrixes_filename":
    os.path.join(base_path_to_model, "matrixes_imagenet_img.pickle"),
    "parameters": {"rect": (256, 256),
                   "crop_size_sx": 227,
                   "crop_size_sy": 227,
                   "visualize": True,
                   "with_affine_transform": False,
                   "number_of_iterations": 10,
                   "number_of_octaves": 4,
                   "octave_scale": 1.4,
                   "clip": True,
                   "end_unit_name": "fc_linear7_forward",
                   "end_gd": "gd_fc_linear7_backward",
                   "step_size": 1.5,
                   "jitter": 32,
                   "affine_scale": 0.05,
                   "affine_iterations": 100}})