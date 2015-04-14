#!/usr/bin/python3
# encoding: utf-8
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 21, 2014

This script to work with Fake Imagenet dataset.

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
import logging
import numpy
import pickle
import os
import scipy.misc
import veles.config as config
from veles.znicz.tests.research.imagenet.processor import Processor

IMAGENET_BASE_PATH = os.path.join(config.root.common.test_dataset_root,
                                  "FakeImagenet/Caffe")


class Main(Processor):

    def __init__(self, **kwargs):
        super(Main, self).__init__(**kwargs)
        self.rect = kwargs.get("rect", (227, 227))
        self.s_mean = numpy.zeros(self.rect + (3,), dtype=numpy.float64)
        self.s_count = numpy.ones_like(self.s_mean)
        self.matrixes = []
        self.s_min = numpy.empty_like(self.s_mean)
        self.s_min[:] = 255
        self.s_max = numpy.zeros_like(self.s_mean)

    def save_images_to_file(self):
        original_data_dir = os.path.join(
            IMAGENET_BASE_PATH, "fake_original_data.dat")
        original_labels_dir = os.path.join(
            IMAGENET_BASE_PATH, "fake_original_labels.pickle")
        count_samples_dir = os.path.join(
            IMAGENET_BASE_PATH, "fake_count_samples.json")
        matrix_dir = os.path.join(IMAGENET_BASE_PATH, "fake_matrixes.pickle")
        train_file = os.path.join(IMAGENET_BASE_PATH, "train.txt")
        valid_file = os.path.join(IMAGENET_BASE_PATH, "valid.txt")
        self.f_samples = open(original_data_dir, "wb")
        original_labels = []
        test_count = 0
        validation_count = 0
        train_count = 0
        labels_count = 0
        for root_path, _tmp, files in os.walk(IMAGENET_BASE_PATH,
                                              followlinks=True):
            for f in files:
                if os.path.splitext(f)[1] == ".JPEG":
                    image_fnme = os.path.join(root_path, f)
                    image = self.decode_image(image_fnme)
                    set_type = root_path[root_path.rfind("/") + 1:]
                    if set_type == "train":
                        continue
                    validation_count += 1
                    image.tofile(self.f_samples)
                    tail = "%s/%s" % (set_type, f)
                    head = image_fnme[:image_fnme.find(tail) - 1]
                    label = head[head.rfind("/") + 1:]
                    tail_to_file = image_fnme[image_fnme.find("Caffe") + 6:]
                    original_labels.append(label)
                    labels_count += 1
                    txt_valid = open(valid_file, "a")
                    txt_valid.write("%s\t%s\n" % (tail_to_file, label))
                    txt_valid.close()
        for root_path, _tmp, files in os.walk(IMAGENET_BASE_PATH,
                                              followlinks=True):
            for f in files:
                if os.path.splitext(f)[1] == ".JPEG":
                    image_fnme = os.path.join(root_path, f)
                    image = self.decode_image(image_fnme)
                    set_type = root_path[root_path.rfind("/") + 1:]
                    if set_type == "validation":
                        continue
                    self.s_mean[:] += image
                    self.s_count[:] += 1.0
                    numpy.minimum(self.s_min[:], image, self.s_min[:])
                    numpy.maximum(self.s_max[:], image, self.s_max[:])
                    train_count += 1
                    image.tofile(self.f_samples)
                    tail = "%s/%s" % (set_type, f)
                    head = image_fnme[:image_fnme.find(tail) - 1]
                    label = head[head.rfind("/") + 1:]
                    tail_to_file = image_fnme[image_fnme.find("Caffe") + 6:]
                    original_labels.append(label)
                    labels_count += 1
                    txt_train = open(train_file, "a")
                    txt_train.write("%s\t%s\n" % (tail_to_file, label))
                    txt_train.close()
        self.count_samples = [test_count, validation_count, train_count]
        self.s_mean /= self.s_count
        mean = numpy.round(self.s_mean)
        numpy.clip(mean, 0, 255, mean)
        mean = mean.astype(numpy.uint8)
        disp = self.s_max - self.s_min
        rdisp = numpy.ones_like(disp.ravel())
        nz = numpy.nonzero(disp.ravel())
        rdisp[nz] = numpy.reciprocal(disp.ravel()[nz])
        rdisp.shape = disp.shape
        out_path_mean = os.path.join(IMAGENET_BASE_PATH, "mean_image.JPEG")
        scipy.misc.imsave(out_path_mean, self.s_mean)
        self.matrixes.append(mean)
        self.matrixes.append(rdisp)
        self.f_samples.close()
        assert labels_count == train_count + validation_count + test_count
        with open(original_labels_dir, "wb") as fout:
            logging.info("Saving labels of images to %s" %
                         original_labels_dir)
            pickle.dump(original_labels, fout)
        with open(count_samples_dir, "w") as fout:
            logging.info("Saving count of test, validation and train to %s"
                         % count_samples_dir)
            json.dump(self.count_samples, fout)
        with open(matrix_dir, "wb") as fout:
            logging.info("Saving mean, min and max matrix to %s" % matrix_dir)
            pickle.dump(self.matrixes, fout)

    def run(self):
        self.save_images_to_file()
        logging.info("End of job")


if __name__ == "__main__":
    main = Main()
    main.run()
