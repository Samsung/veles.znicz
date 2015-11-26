# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on September 8, 2015

Model created for object recognition. Dataset - Imagenet.
Model - convolutional neural network, dynamically
constructed, with pretraining of all layers one by one with autoencoder.

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


import json
import pickle
import os

import numpy
from zope.interface import implementer

from veles.config import root
import veles.error as error
from veles.memory import Array
import veles.opencl_types as opencl_types
import veles.loader as loader


@implementer(loader.ILoader)
class ImagenetLoaderBase(loader.Loader):
    MAPPING = "imagenet_loader_base"
    """loads imagenet from samples.dat, labels.pickle"""
    def __init__(self, workflow, **kwargs):
        super(ImagenetLoaderBase, self).__init__(workflow, **kwargs)
        self.mean = Array()
        self.rdisp = Array()
        self._file_samples_ = ""
        self.sx = kwargs.get("sx", 256)
        self.sy = kwargs.get("sy", 256)
        self.channels = kwargs.get("channels", 3)
        self.original_labels_filename = kwargs.get("original_labels_filename")
        self.count_samples_filename = kwargs.get("count_samples_filename")
        self.matrixes_filename = kwargs.get("matrixes_filename")
        self.samples_filename = kwargs.get("samples_filename")
        self.final_sy = self.sy
        self.final_sx = self.sx

    def initialize(self, **kwargs):
        self._original_labels_ = []
        super(ImagenetLoaderBase, self).initialize(**kwargs)
        self.minibatch_labels.reset(numpy.zeros(
            self.max_minibatch_size, dtype=numpy.int32))

    def load_data(self):
        if (self.original_labels_filename is None or
                not os.path.exists(self.original_labels_filename)):
            raise OSError(
                "original_labels_filename %s does not exist or None."
                " Please specify path to file with labels. If you don't have "
                "pickle with labels, generate it with preparation_imagenet.py"
                % self.original_labels_filename)
        if (self.count_samples_filename is None or
                not os.path.exists(self.count_samples_filename)):
            raise OSError(
                "count_samples_filename %s does not exist or None. Please "
                "specify path to file with count of samples. If you don't "
                "have json file with count of samples, generate it with "
                "preparation_imagenet.py" % self.count_samples_filename)
        if (self.samples_filename is None or
                not os.path.exists(self.samples_filename)):
            raise OSError(
                "samples_filename %s does not exist or None. Please "
                "specify path to file with samples. If you don't "
                "have dat file with samples, generate it with "
                "preparation_imagenet.py" % self.samples_filename)
        with open(self.original_labels_filename, "rb") as fin:
            for lbls in pickle.load(fin):
                txt_lbl, int_lbl = lbls
                self._original_labels_.append(txt_lbl)
                self.labels_mapping[txt_lbl] = int(int_lbl)

        for _ in range(len(self.labels_mapping)):
            self.reversed_labels_mapping.append(None)
        for key, val in self.labels_mapping.items():
            self.reversed_labels_mapping[val] = key

        with open(self.count_samples_filename, "r") as fin:
            for key, value in (json.load(fin)).items():
                set_type = {"test": 0, "val": 1, "train": 2}
                self.class_lengths[set_type[key]] = value
        self.info("Class Lengths: %s", str(self.class_lengths))

        if self.total_samples != len(self._original_labels_):
            raise error.Bug(
                "Number of labels missmatches sum of class lengths")

        self._file_samples_ = open(self.samples_filename, "rb")

        number_of_samples = (self._file_samples_.seek(0, 2) //
                             (self.sx * self.sy * self.channels))

        if number_of_samples != len(self._original_labels_):
            raise error.Bug(
                "Wrong data file size: %s (original data) != %s (original "
                "labels)" % (number_of_samples, len(self._original_labels_)))

    def load_mean(self):
        with open(self.matrixes_filename, "rb") as fin:
            matrixes = pickle.load(fin)
        self.mean.mem = matrixes[0]
        self.rdisp.mem = matrixes[1].astype(
            opencl_types.dtypes[root.common.engine.precision_type])

        if numpy.count_nonzero(numpy.isnan(self.rdisp.mem)):
            raise ValueError("rdisp matrix has NaNs")
        if numpy.count_nonzero(numpy.isinf(self.rdisp.mem)):
            raise ValueError("rdisp matrix has Infs")
        if self.mean.shape != self.rdisp.shape:
            raise ValueError("mean.shape != rdisp.shape")
        if self.mean.shape[0] != self.sy or self.mean.shape[1] != self.sx:
            raise ValueError("mean.shape != (%d, %d)" % (self.sy, self.sx))
        self.has_mean_file = True

    def create_minibatch_data(self):
        sh = [self.max_minibatch_size]
        sh.extend((self.final_sy, self.final_sx, self.channels))
        dtype = opencl_types.dtypes[root.common.engine.precision_type]
        self.minibatch_data.mem = numpy.zeros(sh, dtype=dtype)

    def fill_data(self, index, index_sample, sample):
        self._file_samples_.readinto(self.minibatch_data.mem[index])
        self.minibatch_labels.mem[index] = self.labels_mapping[
            self._original_labels_[int(index_sample)]]

    def fill_indices(self, start_offset, count):
        self.minibatch_indices.map_invalidate()
        idxs = self.minibatch_indices.mem
        self.shuffled_indices.map_read()
        idxs[:count] = self.shuffled_indices[start_offset:start_offset + count]

        if self.is_master:
            return True

        self.minibatch_data.map_invalidate()
        self.minibatch_labels.map_invalidate()

        sample = numpy.zeros(
            [self.sy, self.sx, self.channels], dtype=numpy.uint8)
        sample_bytes = sample.nbytes

        for index, index_sample in enumerate(idxs[:count]):
            self._file_samples_.seek(int(index_sample) * sample_bytes)
            self.fill_data(index, index_sample, sample)

        if count < len(idxs):
            idxs[count:] = self.class_lengths[1]  # no data sample is there
            self.minibatch_data.mem[count:] = numpy.zeros(
                [self.final_sy, self.final_sx, self.channels],
                dtype=numpy.uint8)
            self.minibatch_labels.mem[count:] = 0  # 0 is no data

        return True

    def fill_minibatch(self):
        # minibatch was filled in fill_indices, so fill_minibatch not need
        raise error.Bug("Control should not go here")
