"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on April 1, 2015

Will test correctness and speed of LMDB Loader.

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


import logging
import os
import time

from veles.config import root
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.loader.loader_lmdb import LMDBLoader


@assign_backend("numpy")
class TestLMDBLoader(AcceleratedTest):
    def lmdb_speed(self, kwargs):
        loader = LMDBLoader(self.parent, **kwargs)
        kwargs["snapshot"] = False
        loader.initialize(**kwargs)
        now = time.time()
        while not loader.train_ended:
            loader.run()
        end_run = time.time()
        total_time = end_run - now
        self.info(total_time)

    def test_lmdb_speed_without_cache(self):
        self.info("Will test speed of LMDB without cache")
        kwargs = self.get_kwargs()
        kwargs["use_cache"] = False
        self.lmdb_speed(kwargs)

    def test_lmdb_speed_with_cache(self):
        self.info("Will test speed of LMDB with cache")
        kwargs = self.get_kwargs()
        kwargs["use_cache"] = True
        self.lmdb_speed(kwargs)

    def get_kwargs(self):
        data_path = os.path.join(
            root.common.test_dataset_root, "AlexNet/LMDB_old")
        return {"minibatch_size": 256, "shuffle_limit": 1, "crop": (227, 227),
                "mirror": "random", "color_space": "RGB",
                "normalization_type": "external_mean",
                "train_path": os.path.join(data_path, "ilsvrc12_train_lmdb"),
                "validation_path":
                os.path.join(data_path, "ilsvrc12_val_lmdb"),
                "normalization_parameters": {
                    "mean_source":
                    os.path.join(root.common.test_dataset_root,
                                 "AlexNet/mean_image_227.JPEG")}}


if __name__ == "__main__":
    logging.getLogger("LMDBLoader").setLevel(logging.INFO)
    AcceleratedTest.main()
