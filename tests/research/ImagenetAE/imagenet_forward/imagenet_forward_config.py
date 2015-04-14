# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 17, 2014

Config for forward propagation of Imagenet Model.

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


import numpy

from veles.config import root

root.imagenet_forward.update({
    "loader": {"year": "DET_dataset",
               "series": "DET",
               "path": "/data/veles/datasets/ImagenetAE",
               "path_to_bboxes":
               "/data/veles/datasets/ImagenetAE/raw_bboxes/"
               "raw_bboxes_det_test_npics_40152.4.pickle",
               # "/data/veles/datasets/ImagenetAE/final_jsons
               # /raw_json/need_final_good/"
               # "result_final_11762_12622_raw_DET_dataset_DET_test_1.json",
               "min_index": 0,
               "max_index": 0,
               "angle_step_final": numpy.pi / 12,
               "max_angle_final": numpy.pi / 12,
               "min_angle_final": (-numpy.pi / 12),
               "angle_step_merge": 1,
               "max_angle_merge": 0,
               "min_angle_merge": 0,
               # "angle_step_merge": numpy.pi / 12,
               # "max_angle_merge": numpy.pi / 12,
               # "min_angle_merge": (-numpy.pi / 12),
               "minibatch_size": 32,
               "only_this_file": "000000",
               "raw_bboxes_min_area": 256,
               "raw_bboxes_min_size": 8,
               "raw_bboxes_min_area_ratio": 0.005,
               "raw_bboxes_min_size_ratio": 0.05},
    "trained_workflow": "/data/veles/datasets/ImagenetAE/snapshots/DET/2014/"
                        "imagenet_ae_2014_56.32pt.4.pickle",
    "imagenet_base": "/data/veles/datasets/ImagenetAE",
    "result_path": "/data/veles/tmp/ImagenetAE/final/"
                   "result_final_%d_%d_%s_%s_test_1.json",
    "mergebboxes": {"raw_path":
                        "/data/veles/tmp/ImagenetAE/"
                        "result_raw_final_%d_%d_%s_%s_1.%d.pickle",
                    "ignore_negative": False,
                    "max_per_class": 6,
                    "probability_threshold": 0.45,
                    "last_chance_probability_threshold": 0.39,
                    "mode": "",
                    "labels_compatibility":
                        '/data/veles/datasets/ImagenetAE/temp/216_pool/'
                        'label_compatibility.4.pickle',
                    "use_compatibility": True}
})

# root.imagenet_forward.result_path = root.imagenet_forward.result_path % (
# root.imagenet_forward.loader.min_index,
#     root.imagenet_forward.loader.max_index,
#     root.imagenet_forward.loader.year,
#     root.imagenet_forward.loader.series)
