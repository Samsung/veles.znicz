# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Dec 4, 2014

Wine Loader file.

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
from zope.interface import implementer

import veles.loader as loader


@implementer(loader.IFullBatchLoader)
class WineLoader(loader.FullBatchLoader):
    """Loads Wine dataset.
    """
    MAPPING = "wine_loader"

    def __init__(self, workflow, **kwargs):
        kwargs["normalization_type"] = "pointwise"
        super(WineLoader, self).__init__(workflow, **kwargs)
        self.dataset_file = kwargs["dataset_file"]

    def load_data(self):
        arr = numpy.loadtxt(self.dataset_file, delimiter=',',
                            dtype=numpy.float32)
        self.original_data.mem = arr[:, 1:]
        self.original_labels[:] = arr[:, 0].ravel().astype(numpy.int32) - 1

        self.class_lengths[0] = self.class_lengths[1] = 0
        self.class_lengths[2] = self.original_data.shape[0]
