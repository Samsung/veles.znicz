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

Loads STL-10 dataset binary files.

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
from itertools import repeat

import os
import numpy
from zope.interface import implementer

from veles.loader import IImageLoader, TRAIN, VALID, TEST
from veles.loader.fullbatch_image import FullBatchImageLoader


@implementer(IImageLoader)
class STL10FullBatchLoader(FullBatchImageLoader):
    MAPPING = "full_batch_stl_10"
    SIZE = 96, 96
    SQUARE = SIZE[0] * SIZE[1] * 3

    def __init__(self, workflow, **kwargs):
        super(STL10FullBatchLoader, self).__init__(workflow, **kwargs)
        self._directory = kwargs["directory"]
        self.original_shape = self.SIZE

    def init_unpickled(self):
        super(STL10FullBatchLoader, self).init_unpickled()
        self._files_ = [None] * 3

    def __del__(self):
        for file in self._files_:
            if file is not None:
                file.close()

    def initialize(self, device, **kwargs):
        self.directory = self._directory
        super(STL10FullBatchLoader, self).initialize(device=device, **kwargs)

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, value):
        if not os.path.isdir(value):
            raise ValueError("\"%s\" must be a directory" % value)
        self._directory = value
        with open(os.path.join(self.directory, "class_names.txt"), 'r') as fin:
            self._class_names = fin.read().split()
        for file in self._files_:
            if file is not None:
                file.close()
        self._files_[TRAIN] = open(
            os.path.join(self.directory, "train_X.bin"), "rb")
        self._files_[VALID] = open(
            os.path.join(self.directory, "test_X.bin"), "rb")
        self._labels = [None] * 3
        self._labels[TRAIN] = numpy.fromfile(
            os.path.join(self.directory, "train_y.bin"), dtype=numpy.uint8)
        self._labels[VALID] = numpy.fromfile(
            os.path.join(self.directory, "test_y.bin"), dtype=numpy.uint8)
        for klass in TRAIN, VALID:
            file = self._files_[klass]
            file.seek(0, os.SEEK_END)
            assert file.tell() / self.SQUARE == len(self._labels[klass])

    def get_image_label(self, key):
        return self._class_names[self._labels[key[0]][key[1]] - 1]

    def get_image_info(self, key):
        return self.SIZE, "RGB"

    def get_image_data(self, key):
        file = self._files_[key[0]]
        file.seek(key[1] * self.SQUARE, os.SEEK_SET)
        return numpy.transpose(
            numpy.frombuffer(file.read(self.SQUARE), dtype=numpy.uint8)
            .reshape((3,) + self.SIZE), (1, 2, 0))

    def get_keys(self, index):
        if index == TEST:
            return []
        file = self._files_[index]
        file.seek(0, os.SEEK_END)
        return zip(repeat(index), range(file.tell() // self.SQUARE))
