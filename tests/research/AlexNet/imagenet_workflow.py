# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Dec 19, 2014

Model created for object recognition. Database - Imagenet (1000 classes).
Self-constructing Model. It means that Model can change for any Model
(Convolutional, Fully connected, different parameters) in configuration file.

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


import cv2
import numpy
import os
from veles.loader.interactive import InteractiveLoader

from veles.config import root
import veles.prng as prng
from veles.znicz.standard_workflow import StandardWorkflow

from veles.znicz.loader.imagenet_loader import ImagenetLoaderBase
from veles.znicz.loader import loader_lmdb  # pylint: disable=W0611


root.common.ThreadPool.maxthreads = 3


class ImagenetLoader(ImagenetLoaderBase):
    MAPPING = "imagenet_pickle_loader"

    def __init__(self, workflow, **kwargs):
        super(ImagenetLoader, self).__init__(workflow, **kwargs)
        self.crop_size_sx = kwargs.get("crop_size_sx", 227)
        self.crop_size_sy = kwargs.get("crop_size_sy", 227)
        self.do_mirror = False
        self.mirror = kwargs.get("mirror", False)
        self.final_sy = self.crop_size_sy
        self.final_sx = self.crop_size_sx
        self.has_mean_file = False

    def load_data(self):
        super(ImagenetLoader, self).load_data()
        if self.matrixes_filename and os.path.exists(self.matrixes_filename):
            self.load_mean()
            self.has_mean_file = True

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

    def fill_data(self, index, index_sample, sample):
        self._file_samples_.readinto(sample)
        rand = prng.get()
        if self.minibatch_class == 2:
            self.do_mirror = self.mirror and bool(rand.randint((2)))
        else:
            self.do_mirror = False
        image = self.transform_sample(sample)
        self.minibatch_data.mem[index] = image
        self.minibatch_labels.mem[
            index] = self.labels_mapping[
            self._original_labels_[int(index_sample)]]


class InteractiveImagenetLoader(InteractiveLoader, ImagenetLoader):
    MAPPING = "interactive_imagenet"
    DISABLE_INTERFACE_VERIFICATION = True

    def derive_from(self, loader):
        super(InteractiveImagenetLoader, self).derive_from(loader)

        self.do_mirror = False
        self.has_mean_file = True
        if self.matrixes_filename and os.path.exists(self.matrixes_filename):
            self.load_mean()

    def load_data(self):
        InteractiveLoader.load_data(self)

    def create_minibatch_data(self):
        InteractiveLoader.create_minibatch_data(self)

    def fill_minibatch(self):
        InteractiveLoader.fill_minibatch(self)

    def _feed(self, data):
        data = data[0]
        data = cv2.resize(
            data, (self.sx, self.sy), interpolation=cv2.INTER_LANCZOS4)
        data = self.transform_sample(data)
        self.minibatch_data.mem[0] = data

    def fill_indices(self, start_offset, count):
        for v in (self.minibatch_data, self.minibatch_labels,
                  self.minibatch_indices):
            v.map_invalidate()
        self.shuffled_indices.map_read()
        self.minibatch_indices.mem[:count] = self.shuffled_indices[
            start_offset:start_offset + count]
        return False


class ImagenetWorkflow(StandardWorkflow):
    """
    Imagenet Workflow
    """

    def create_workflow(self):
        self.link_repeater(self.start_point)
        self.link_loader(self.repeater)
        self.link_forwards(("input", "minibatch_data"), self.loader)
        self.link_evaluator(self.forwards[-1])
        self.link_decision(self.evaluator)
        self.link_snapshotter(self.decision)
        parallel_units = []
        # self.link_image_saver(self.snapshotter)
        if root.imagenet.add_plotters:
            parallel_units.extend(link(self.snapshotter) for link in (
                self.link_error_plotter,
                self.link_err_y_plotter))
            parallel_units.append(self.link_weights_plotter(
                "weights", self.snapshotter))

        last_gd = self.link_gds(*parallel_units)
        self.link_lr_adjuster(last_gd)
        self.link_loop(self.lr_adjuster)
        self.link_end_point(self.lr_adjuster)


def run(load, main):
    load(ImagenetWorkflow,
         loader_name=root.imagenet.loader_name,
         loader_config=root.imagenet.loader,
         decision_config=root.imagenet.decision,
         snapshotter_config=root.imagenet.snapshotter,
         weights_plotter_config=root.imagenet.weights_plotter,
         lr_adjuster_config=root.imagenet.lr_adjuster,
         layers=root.imagenet.layers,
         image_saver_config=root.imagenet.image_saver,
         loss_function=root.imagenet.loss_function)
    main()
