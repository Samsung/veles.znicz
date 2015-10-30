# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Dec 8, 2014

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


root.yalefaces.update({
    "preprocessing": True,
    "datasaver": {"file_name":
                  os.path.join(root.common.dirs.datasets,
                               "yale_faces_minibatches.sav")},
    "loader_name": "full_batch_auto_label_file_image",
    "loader": {"minibatch_size": 1, "force_numpy": False,
               "validation_ratio": 0.15,
               "file_subtypes": ["x-portable-graymap"],
               "ignored_files": [".*Ambient.*"],
               "shuffle_limit": 0,
               "add_sobel": False,
               "normalization_type": "mean_disp",
               "mirror": False,
               "color_space": "GRAY",
               "background_color": (0,),
               "train_paths":
               [os.path.join(root.common.dirs.datasets, "CroppedYale")]}})
