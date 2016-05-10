# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Feb 15, 2016

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
import cv2
import pickle
import numpy
import os
import shutil

import veles
from veles.config import root
from veles.downloader import Downloader
from veles.loader import Loader
from veles.loader.file_image import FileListImageLoader
from veles.memory import Array
import veles.opencl_types as opencl_types
from imagenet_workflow import ImagenetPreprocessingBase


class ImagenetTestLoader(FileListImageLoader, ImagenetPreprocessingBase):
    def __init__(self, workflow, **kwargs):
        super(ImagenetTestLoader, self).__init__(workflow, **kwargs)
        self.sx = kwargs.get("sx", 256)
        self.sy = kwargs.get("sy", 256)
        self.channels = kwargs.get("channels", 3)
        self.mean = Array()
        self.rdisp = Array()
        self.matrixes_filename = kwargs.get("matrixes_filename")
        if self.matrixes_filename and os.path.exists(self.matrixes_filename):
            self.load_mean()
            self.has_mean_file = True

    @Loader.shape.getter
    def shape(self):
        shape = (self.crop_size_sx, self.crop_size_sy, self.channels)
        return shape

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

    def _load_image(self, key, crop=True):
        data = self.get_image_data(key)
        size, color = self.get_image_info(key)

        if color != self.color_space:
            method = getattr(
                cv2, "COLOR_%s2%s" % (color, self.color_space), None)
            if method is None:
                aux_method = getattr(cv2, "COLOR_%s2BGR" % color)
                try:
                    data = cv2.cvtColor(data, aux_method)
                except cv2.error as e:
                    raise AttributeError(
                        "Failed to perform '%s' conversion", aux_method)
                method = getattr(cv2, "COLOR_BGR2%s" % self.color_space)
            try:
                data = cv2.cvtColor(data, method)
            except cv2.error as e:
                raise AttributeError(
                    "Failed to perform '%s' conversion", aux_method)

        bbox = self.get_image_bbox(key, size)
        sample = cv2.resize(
            data, (self.sx, self.sy),
            interpolation=cv2.INTER_LANCZOS4)
        return self.transform_sample(sample), 1, bbox


def create_forward(workflow, normalizer, labels_mapping, loader_config):
    # Disable plotters:
    workflow.plotters_are_enabled = False

    # Link downloader
    workflow.start_point.unlink_after()
    workflow.downloader = Downloader(
        workflow,
        url="https://s3-eu-west-1.amazonaws.com/veles.forge/"
            "AlexNet/imagenet_test.tar",
        directory=root.common.dirs.datasets,
        files=["imagenet_test"])
    workflow.downloader.link_from(workflow.start_point)
    workflow.repeater.link_from(workflow.downloader)

    # Cnanging Imagenet Loader to another Loader:
    new_loader = workflow.change_unit(
        workflow.loader.name,
        ImagenetTestLoader(workflow, **loader_config))

    workflow.loader = new_loader

    # Link attributes:
    workflow.forwards[0].link_attrs(
        new_loader, ("input", "minibatch_data"))

    workflow.evaluator.link_attrs(
        new_loader,
        ("batch_size", "minibatch_size"),
        ("labels", "minibatch_labels"),
        ("max_samples_per_epoch", "total_samples"),
        "class_lengths", ("offset", "minibatch_offset"))
    workflow.decision.link_attrs(
        new_loader, "minibatch_class", "last_minibatch",
        "minibatch_size", "class_lengths", "epoch_ended", "epoch_number")

    workflow.evaluator.link_attrs(new_loader, "class_keys")

    # Set normalizer from previous Loader to new one:
    new_loader._normalizer = normalizer

    # Set labels_mapping and class_keys in Evaluator to correct writting the
    # results:
    workflow.evaluator.labels_mapping = labels_mapping


if __name__ == "__main__":
    parameters = {
        "dry_run": "init",
        "snapshot":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/AlexNet/"
        "imagenet_test_82.81_validation_40.68_train_32.10.4.pickle.gz",
        "stealth": True,
        "device": 0}

    path_mod = "veles/znicz/tests/research/AlexNet/imagenet_workflow.py"
    path_con = \
        "veles/znicz/tests/research/AlexNet/imagenet_workflow_nin_config.py"
    data_path = os.path.join(root.common.dirs.datasets, "imagenet_test")
    path_to_labeled_frames = os.path.join(data_path, "out_pictures")

    # Load workflow from snapshot
    launcher = veles(path_mod, path_con, **parameters)  # pylint: disable=E1102

    # Swith to testing mode:
    launcher.testing = True
    loader_conf = {"minibatch_size": 1,
                   "shuffle_limit": 0,
                   "matrixes_filename": os.path.join(
                       data_path, "matrixes_imagenet_img.pickle"),
                   "sx": 256,
                   "sy": 256,
                   "crop_size_sx": 224,
                   "crop_size_sy": 224,
                   "channels": 3,
                   "crop": (1, 1),
                   "normalization_type": "none",
                   "scale": (256, 256),
                   "class_keys_path": os.path.join(
                       data_path, "class_keys_imagenet_img.json"),
                   "base_directory":
                   os.path.join(data_path, "input_pictures"),
                   "path_to_test_text_file":
                   [os.path.join(data_path, "imagenet_test.txt")]}
    create_forward(
        launcher.workflow, normalizer=launcher.workflow.loader.normalizer,
        labels_mapping=launcher.workflow.loader.reversed_labels_mapping,
        loader_config=loader_conf)

    if os.path.exists(path_to_labeled_frames):
        shutil.rmtree(path_to_labeled_frames)

    # Initialize and run new workflow:
    launcher.boot()

    os.makedirs(path_to_labeled_frames)

    # Write results:
    results = launcher.workflow.gather_results()

    output = results["Output"]

    for path_to_original, label in output.items():
        image = cv2.imread(path_to_original)

        font = cv2.FONT_HERSHEY_DUPLEX

        name_image = os.path.basename(path_to_original).split(".")[0]
        new_path = os.path.join(
            path_to_labeled_frames, name_image + "_%s" % label + ".png")
        print("Saved image to %s" % new_path)
        cv2.imwrite(new_path, image)

    out_file = os.path.join(data_path, "result.txt")
    with open(out_file, "w") as fout:
        json.dump(results, fout, sort_keys=True)
    print("Successfully wrote %d results to %s" % (len(results), out_file))
