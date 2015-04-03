# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Nov 14, 2014

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

import cv2
import numpy
from zope.interface import implementer

from veles.config import root
import veles.error as error
from veles.memory import Vector
import veles.opencl_types as opencl_types
import veles.loader as loader
import veles.prng.random_generator as prng


@implementer(loader.ILoader)
class ImagenetLoader(loader.Loader):
    """loads imagenet from samples.dat, labels.pickle"""
    MAPPING = "imagenet_pickle_loader"

    def __init__(self, workflow, **kwargs):
        super(ImagenetLoader, self).__init__(workflow, **kwargs)
        self.mean = Vector()
        self.rdisp = Vector()
        self.file_samples = ""
        self.crop_size_sx = kwargs.get("crop_size_sx", 227)
        self.crop_size_sy = kwargs.get("crop_size_sy", 227)
        self.sx = kwargs.get("sx", 256)
        self.sy = kwargs.get("sy", 256)
        self.shuffle_limit = kwargs.get("shuffle_limit", 2000000000)
        self.original_labels_filename = kwargs.get(
            "original_labels_filename", None)
        self.count_samples_filename = kwargs.get(
            "count_samples_filename", None)
        self.matrixes_filename = kwargs.get("matrixes_filename", None)
        self.samples_filename = kwargs.get("samples_filename", None)
        self.has_mean_file = False
        self.do_mirror = False
        self.mirror = kwargs.get("mirror", False)
        self.channels = kwargs.get("channels", 3)

    def init_unpickled(self):
        super(ImagenetLoader, self).init_unpickled()
        self.original_labels = None

    def __getstate__(self):
        state = super(ImagenetLoader, self).__getstate__()
        state["original_labels"] = None
        state["file_samples"] = None
        return state

    def initialize(self, **kwargs):
        self.normalizer.reset()
        super(ImagenetLoader, self).initialize(**kwargs)
        self.minibatch_labels.reset(numpy.zeros(
            self.max_minibatch_size, dtype=numpy.int32))

    def shuffle(self):
        if self.shuffle_limit <= 0:
            return
        self.shuffle_limit -= 1
        self.info("Shuffling, remaining limit is %d", self.shuffle_limit)
        super(ImagenetLoader, self).shuffle()

    def load_data(self):
        self.original_labels = []

        with open(self.original_labels_filename, "rb") as fin:
            for lbl in pickle.load(fin):
                self.original_labels.append(int(lbl))
                self.labels_mapping[int(lbl)] = int(lbl)
        self.info("Labels (min max count): %d %d %d",
                  numpy.min(self.original_labels),
                  numpy.max(self.original_labels),
                  len(self.original_labels))

        with open(self.count_samples_filename, "r") as fin:
            for key, value in (json.load(fin)).items():
                set_type = {"test": 0, "val": 1, "train": 2}
                self.class_lengths[set_type[key]] = value
        self.info("Class Lengths: %s", str(self.class_lengths))

        if self.total_samples != len(self.original_labels):
            raise error.Bug(
                "Number of labels missmatches sum of class lengths")

        with open(self.matrixes_filename, "rb") as fin:
            matrixes = pickle.load(fin)

        self.mean.mem = matrixes[0]
        self.rdisp.mem = matrixes[1].astype(
            opencl_types.dtypes[root.common.precision_type])
        if numpy.count_nonzero(numpy.isnan(self.rdisp.mem)):
            raise ValueError("rdisp matrix has NaNs")
        if numpy.count_nonzero(numpy.isinf(self.rdisp.mem)):
            raise ValueError("rdisp matrix has Infs")
        if self.mean.shape != self.rdisp.shape:
            raise ValueError("mean.shape != rdisp.shape")
        if self.mean.shape[0] != self.sy or self.mean.shape[1] != self.sx:
            raise ValueError("mean.shape != (%d, %d)" % (self.sy, self.sx))

        self.file_samples = open(self.samples_filename, "rb")
        if (self.file_samples.seek(0, 2)
                // (self.sx * self.sy * self.channels) !=
                len(self.original_labels)):
            raise error.Bug("Wrong data file size")

    def create_minibatch_data(self):
        sh = [self.max_minibatch_size]
        sh.extend((self.crop_size_sy, self.crop_size_sx, self.channels))
        dtype = opencl_types.dtypes[root.common.precision_type]
        self.minibatch_data.mem = numpy.zeros(sh, dtype=dtype)

    def transform_sample(self, sample):
        if self.has_mean_file:
            sample = self.deduct_mean(sample)
        if self.crop_size_sx and self.crop_size_sy:
            sample = self.cut_out(sample)
        if self.do_mirror:
            sample = self.mirror_sample(sample)
        return sample

    def deduct_mean(self, sample):
        sample = sample.astype(self.rdisp.dtype)
        sample -= self.mean.mem
        sample *= self.rdisp.mem
        return sample

    def mirror_sample(self, sample):
        mirror_sample = numpy.zeros_like(sample)
        cv2.flip(sample, 1, mirror_sample)
        return mirror_sample

    def cut_out(self, sample):
        if self.minibatch_class == 2:
            rand = prng.get()
            h_off = rand.randint(
                sample.shape[0] - self.crop_size_sy + 1)
            w_off = rand.randint(
                sample.shape[1] - self.crop_size_sx + 1)
        else:
            h_off = (sample.shape[0] - self.crop_size_sy) / 2
            w_off = (sample.shape[1] - self.crop_size_sx) / 2
        sample = sample[
            h_off:h_off + self.crop_size_sy,
            w_off:w_off + self.crop_size_sx, :self.channels]
        return sample

    def fill_indices(self, start_offset, count):
        self.minibatch_indices.map_invalidate()
        idxs = self.minibatch_indices.mem
        self.shuffled_indices.map_read()
        idxs[:count] = self.shuffled_indices[start_offset:start_offset + count]

        if self.is_master:
            return True

        if self.matrixes_filename is not None:
            self.has_mean_file = True

        self.minibatch_data.map_invalidate()
        self.minibatch_labels.map_invalidate()

        sample_bytes = self.mean.mem.nbytes
        sample = numpy.zeros_like(self.mean.mem, dtype=numpy.uint8)

        for index, index_sample in enumerate(idxs[:count]):
            self.file_samples.seek(int(index_sample) * sample_bytes)
            self.file_samples.readinto(sample)
            rand = prng.get()
            self.do_mirror = self.mirror and bool(rand.randint((2)))
            image = self.transform_sample(sample)
            self.minibatch_data.mem[index] = image
            self.minibatch_labels.mem[
                index] = self.original_labels[int(index_sample)]

        if count < len(idxs):
            idxs[count:] = self.class_lengths[1]  # no data sample is there
            self.croped_mean = self.cut_out(self.mean)
            self.minibatch_data.mem[count:] = self.croped_mean
            self.minibatch_labels.mem[count:] = 0  # 0 is no data

        return True

    def fill_minibatch(self):
        # minibatch was filled in fill_indices, so fill_minibatch not need
        raise error.Bug("Control should not go here")
