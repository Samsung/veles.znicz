# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 20, 2013

ImageSaver unit.

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


from __future__ import division

import glob
import numpy
import os
from PIL import Image
from zope.interface import implementer

import veles.config as config
from veles.error import BadFormatError
from veles.distributable import TriviallyDistributable
from veles.units import Unit, IUnit


@implementer(IUnit)
class ImageSaver(Unit, TriviallyDistributable):
    """Saves input to pngs in the supplied directory.

    Will remove all existing png files in the supplied directory.

    Attributes:
        out_dirs: output directories by minibatch_class where to save png.
        input: batch with input samples.
        output: batch with corresponding output samples (may be None).
        target: batch with corresponding target samples (may be None).
        indices: sample indices.
        labels: sample labels.
        max_idx: indices of element with maximum value for each sample.

    Remarks:
        if max_idx is not None:
            Softmax classifier is assumed and only failed samples
            will be saved.
        else:
            MSE task is assumed and output and target
            should be None or not None both simultaneously.
    """
    def __init__(self, workflow, **kwargs):
        super(ImageSaver, self).__init__(workflow, **kwargs)
        self.out_dirs = kwargs.get(
            "out_dirs", [os.path.join(config.root.common.dirs.cache,
                                      "tmpimg/test"),
                         os.path.join(config.root.common.dirs.cache,
                                      "tmpimg/validation"),
                         os.path.join(config.root.common.dirs.cache,
                                      "tmpimg/train")])
        self.limit = kwargs.get("limit", 100)
        self.output = None  # memory.Array()
        self.target = None  # memory.Array()
        self.max_idx = None  # memory.Array()
        self._last_save_time = 0
        self.save_time = 0
        self._n_saved = [0, 0, 0]
        self._remembers_gates = False
        self.color_space = kwargs.get("color_space", "RGB")
        self.demand("input", "indices", "labels",
                    "minibatch_class", "minibatch_size")
        self.reversed_labels_mapping = None

    @staticmethod
    def as_image(inp):
        if len(inp.shape) == 1:
            return None
        elif len(inp.shape) == 2:
            if 1 in inp.shape:
                return None
            return inp.reshape(inp.shape[0], inp.shape[1])
        elif len(inp.shape) == 3:
            if inp.shape[2] == 3:
                return inp
            if inp.shape[0] == 3:
                return inp.transpose(1, 2, 0)
            if inp.shape[2] == 4:
                return inp[:, :, 0:3]
        else:
            raise BadFormatError()

    def initialize(self, **kwargs):
        pass

    def get_list_indices_to_save(self):
        indices_to_save = []
        for image_index in range(self.minibatch_size):
            true_label = self.labels[image_index]
            if self.max_idx is not None:
                prediction_label = self.max_idx[image_index]
                if prediction_label != true_label:
                    indices_to_save.append(image_index)
            else:
                indices_to_save.append(image_index)
        return indices_to_save

    def create_directory(self, dirnme):
        try:
            os.makedirs(dirnme, mode=0o775)
        except OSError:
            pass

    def remove_old_pictures(self):
        if self._last_save_time < self.save_time:
            self._last_save_time = self.save_time

            for i in range(len(self._n_saved)):
                self._n_saved[i] = 0
            for dirnme in self.out_dirs:
                files = glob.glob("%s/*.png" % dirnme)
                for file in files:
                    try:
                        os.unlink(file)
                    except OSError:
                        pass

    def save_image(self, image, path):
        image_to_save = Image.fromarray(image)
        try:
            image_to_save.save(path)
        except ValueError:
            self.warning(
                "Could not save image to %s. Image does not have a suitable"
                "array shape for any mode. Image shape: %s"
                % (path, str(image.shape)))
        except OSError:
            self.warning("Could not save image to %s" % (path))

    def normalize_image(self, image, colorspace=None):
        """Normalizes numpy array to interval [0, 255].
        """
        float_image = image.astype(numpy.float32)
        if float_image.__array_interface__[
                "data"][0] == image.__array_interface__["data"][0]:
            float_image = float_image.copy()
        float_image -= float_image.min()
        max_value = float_image.max()
        if max_value:
            max_value /= 255.0
            float_image /= max_value
        else:
            float_image[:] = 127.5
        normalized_image = float_image.astype(numpy.uint8)
        if (colorspace != "RGB" and len(normalized_image.shape) == 3
                and normalized_image.shape[2] == 3):
            import cv2
            normalized_image = cv2.cvtColor(
                normalized_image, getattr(cv2, "COLOR_" + colorspace + "2RGB"))
        return normalized_image

    def read_data(self):
        for data in (self.output, self.max_idx, self.target):
            if data is not None:
                data.map_read()
        for data in (self.indices, self.input, self.labels):
            data.map_read()

    def run(self):
        self.read_data()
        for dirnme in self.out_dirs:
            self.create_directory(dirnme)
        self.remove_old_pictures()

        if self._n_saved[self.minibatch_class] >= self.limit:
            return

        self.save_images(self.get_list_indices_to_save())

    def get_paths_and_save_image(self, image_index):
        input_image = ImageSaver.as_image(self.input[image_index])
        true_label = self.labels[image_index]
        index = self.indices.mem[image_index]
        if self.reversed_labels_mapping is not None:
            true_label = self.reversed_labels_mapping[true_label]
        if self.max_idx is not None:
            prediction_label = self.max_idx[image_index]
            output_image = self.output[image_index]
            out_path_dir = self.out_dirs[self.minibatch_class]
            if self.reversed_labels_mapping is not None:
                text_prediction_label = self.reversed_labels_mapping[
                    prediction_label]
            else:
                text_prediction_label = prediction_label
            tail_file_name = "%s_as_%s.%.0fpt.%d.png" % (
                true_label,
                text_prediction_label,
                output_image[prediction_label], index)
            target_image = None
        else:
            out_path_dir = os.path.join(
                self.out_dirs[self.minibatch_class], "%d" % index)
            if self.output is not None and self.target is not None:
                output_image, target_image = (
                    ImageSaver.as_image(v[image_index]) for v in
                    (self.output, self.target))

                if output_image is None:
                    assert target_image is None, \
                        "Output shape is %s while target shape is %s" % (
                            self.output.shape[1:], target_image.shape)
                else:
                    output_image = output_image.reshape(target_image.shape)
                mse = numpy.linalg.norm(
                    self.output.mem - self.target.mem) / input_image.size
            else:
                output_image = None
                target_image = None
                mse = None
            if mse is not None:
                tail_file_name = "%.6f_%s_%d.png" % (mse, true_label, index)
            else:
                tail_file_name = "%s_%s_%d.png" % (mse, true_label, index)

        self.create_directory(out_path_dir)

        out_path_input_image, out_path_output_image, out_path_target = (
            os.path.join(out_path_dir, m % tail_file_name) for m in
            ("input_image_%s", "output_image_%s", "target_%s"))

        self.save_image(
            self.normalize_image(input_image, self.color_space),
            out_path_input_image)

        if output_image is not None and self.max_idx is None:
            self.save_image(
                self.normalize_image(output_image, self.color_space),
                out_path_output_image)

        if target_image is not None:
            self.save_image(
                self.normalize_image(target_image, self.color_space),
                out_path_target)

        self._n_saved[self.minibatch_class] += 1

    def save_images(self, indices_to_save):
        for image_index in indices_to_save:
            self.get_paths_and_save_image(image_index)
            if self._n_saved[self.minibatch_class] >= self.limit:
                return
