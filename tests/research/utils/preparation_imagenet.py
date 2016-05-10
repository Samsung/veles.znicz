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

Created on Dec 10, 2014

This script to work with Imagenet dataset. AlexNet workflow works with
"img" files. ImagenetAE workflow works with "DET" files. First, select the
challenge in "series" partameter: "DET" for detection imagenet challenge,
"img" for classification and localization challenge. Second, select folder with
Imagenet dataset in "root_path" parameter. Please make sure,
that parameter "rect" is correct. It is (216, 216) for "DET" challenge
in standard configuration. It is (256, 256) for "img" challenge in standard
configuration. Then, run "init_dataset" command to save in json all
information about dataset. Second, run "save_dataset_to_file" command, which
generates pickles and .dat files for Imagenet Pickle Loader.
If you want to check the result, run "test_load_data" command. You can
prepare validation set with "save_validation_to_forward" command after
training Neural Network. And use it in ImagenetForward workflow, for example.

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
import json
import logging
import matplotlib.pyplot as plt
import numpy
import pickle
from PIL import Image
import os
import scipy.misc
import sys
import time

from veles.compat import from_none
from veles.config import root
import veles.opencl_types as opencl_types
import veles.prng as rnd
from veles.znicz.external import xmltodict
from veles.znicz.tests.research.ImagenetAE.processor import Processor


IMAGES_JSON = "images_imagenet_%s_%s_%s.json"
TEST = "test"
TRAIN = "train"
VALIDATION = "val"

root.prep_imagenet.root_name = "imagenet"
root.prep_imagenet.series = "img"
root.prep_imagenet.root_path = os.path.join(
    root.common.dirs.datasets, "AlexNet", root.prep_imagenet.root_name)

root.prep_imagenet.update({
    "file_with_indices":
    os.path.join(root.prep_imagenet.root_path,
                 "indices_imagenet_2014_list.txt"),
    "file_indices_to_categories":
    os.path.join(root.prep_imagenet.root_path, "indices_to_categories.txt"),
    "file_text_to_int_labels":
    os.path.join(
        root.prep_imagenet.root_path,
        "text_to_int_labels_%s_%s.json"
        % (root.prep_imagenet.root_name, root.prep_imagenet.series)),
    "file_original_labels":
    os.path.join(
        root.prep_imagenet.root_path,
        "original_labels_%s_%s.pickle"
        % (root.prep_imagenet.root_name, root.prep_imagenet.series)),
    "file_original_data":
    os.path.join(
        root.prep_imagenet.root_path,
        "original_data_%s_%s.dat"
        % (root.prep_imagenet.root_name, root.prep_imagenet.series)),
    "file_matrix":
    os.path.join(
        root.prep_imagenet.root_path,
        "matrixes_%s_%s.pickle"
        % (root.prep_imagenet.root_name, root.prep_imagenet.series)),
    "file_hierarchy":
    os.path.join(root.prep_imagenet.root_path,
                 "hierarchy_2014_DET_train_0.json"),
    "file_count_samples":
    os.path.join(
        root.prep_imagenet.root_path,
        "count_samples_%s_%s.json"
        % (root.prep_imagenet.root_name, root.prep_imagenet.series)),
    "rect": (256, 256),
    "channels": 3,
    "get_label": "all_ways",
    # "from_image_name" "from_image_path" "from_xml" "all_ways"
    "get_label_from_txt_label": True,
    "command_to_run": "save_dataset_to_file"
    # "save_dataset_to_file" "init_dataset" "test_load_data"
    # "save_validation_to_forward"
})

root.prep_imagenet.classes_count = (
    200 if root.prep_imagenet.series == "DET" else 1000)

root.prep_imagenet.MAPPING = {
    "% s" % root.prep_imagenet.root_name: {
        "img": {
            "train": ("ILSVRC2012_img_train", "ILSVRC2012_bbox_train_v2"),
            "val": ("ILSVRC2012_img_val", "ILSVRC2012_bbox_val_v3"),
            "test": ("ILSVRC2012_img_test", "")},
        "DET": {
            "train": ("ILSVRC2014_DET_train", "ILSVRC2014_DET_bbox_train"),
            "val": ("ILSVRC2013_DET_val", "ILSVRC2013_DET_bbox_val"),
            "test": ("ILSVRC2013_DET_test", "")}}}


class PreparationImagenet(Processor):
    """
    This script to work with Imagenet dataset. AlexNet workflow works with
"img" files. ImagenetAE workflow works with "DET" files. First, select the
challenge in "series" partameter: "DET" for detection imagenet challenge,
"img" for classification and localization challenge. Second, select folder with
Imagenet dataset in "root_path" parameter. Please make sure,
that parameter "rect" is correct. It is (216, 216) for "DET" challenge
in standard configuration. It is (256, 256) for "img" challenge in standard
configuration. Then, run "init_dataset" command to save in json all
information about dataset. Second, run "save_dataset_to_file" command, which
generates pickles and .dat files for Imagenet Pickle Loader.
If you want tro check the result, run "test_load_data" command. You can
prepare validation set with "save_validation_to_forward" command after
training Neural Network. And use it in ImagenetForward workflow, for example.
    """

    def __init__(self, **kwargs):
        super(PreparationImagenet, self).__init__(**kwargs)
        self.rect = kwargs.get("rect", root.prep_imagenet.rect)
        self.root_name = root.prep_imagenet.root_name
        self.series = root.prep_imagenet.series
        self.root_path = root.prep_imagenet.root_path
        self.images = {TEST: {}, TRAIN: {}, VALIDATION: {}}
        self.s_sum = numpy.zeros(
            self.rect + (root.prep_imagenet.channels,), dtype=numpy.float64)
        self.s_mean = numpy.zeros_like(self.s_sum)
        self.s_count = numpy.ones_like(self.s_mean)
        self.s_min = numpy.empty_like(self.s_mean)
        self.s_min[:] = 255
        self.s_max = numpy.zeros_like(self.s_mean)
        self.k = 0
        self.label_ind = 0
        self.labels_int_txt = {}
        self.map_items = None
        self.labels = []
        self.labels_diff = set()
        self.original_labels = []
        self.command_to_run = root.prep_imagenet.command_to_run
        self.count_samples = {TEST: 0, TRAIN: 0, VALIDATION: 0}
        self.matrixes = []
        self.colorspace = kwargs.get("colorspace", "RGB")
        self._include_derivative = kwargs.get("derivative", True)
        self._sobel_kernel_size = kwargs.get("sobel_kernel_size", 5)
        self.threshold_val = 1
        self.threshold_train = 8
        self.path_to_categories = os.path.join(
            root.prep_imagenet.root_path, "indices_to_categories.txt")
        self.num_word = []
        self.class_keys = {0: [], 1: [], 2: []}

    def initialize(self):
        self.map_items = root.prep_imagenet.MAPPING[
            self.root_name][self.series].items()

    def init_files_img(self):
        self.info("Looking for images in %s:", self.root_path)
        for set_type, (dir_images, dir_bboxes) in sorted(self.map_items):
            path = os.path.join(self.root_path, dir_images)
            self.info("Scanning JPG %s...", path)
            for root_p, _tmp, files in os.walk(path, followlinks=True):
                for f in files:
                    if os.path.splitext(f)[1] == ".JPEG":
                        f_path = os.path.join(root_p, f)
                        self.get_images_from_path(f_path, [], set_type)
                    else:
                        self.warning("Unexpected file in dir %s", f)
        self.info("Saving images to %s" % self.root_path)
        self.save_images_to_json(self.images)
        self.info("Label is None! %s times" % self.k)
        return None

    def get_label_from_image_name(self, image_path):
        image_name = os.path.basename(image_path)
        labels = []
        temp_label = image_name.split('_')[0]
        if temp_label in self.get_labels_list_from_file():
            labels.append(temp_label)
        else:
            if root.prep_imagenet.get_label == "all_ways":
                return labels
            self.warning(
                "Not a label %s in picture %s" % (temp_label, image_name))
        return labels

    def get_label_from_image_path(self, image_path):
        labels = []
        dir_path = os.path.dirname(image_path)
        dir_name = os.path.basename(dir_path)
        temp_label = dir_name
        if temp_label in self.get_labels_list_from_file():
            labels.append(temp_label)
        else:
            if root.prep_imagenet.get_label == "all_ways":
                return labels
            self.warning(
                "Not a label %s in picture %s" % (temp_label, image_path))
        return labels

    def get_label_from_xml(self, image_path):
        result_labels = []
        labels = []
        label = None
        for set_type, (dir_images, dir_bboxes) in (
                i for i in sorted(self.map_items) if i[1][1]):
            if image_path.find(dir_images) == -1:
                continue
            head, image_tail = image_path.split(dir_images)
            xml_path = head + dir_bboxes + image_tail.replace(".JPEG", ".xml")
            if not os.access(xml_path, os.R_OK):
                if xml_path.find("test") < 0:
                    self.warning(
                        "File %s does not exist or read permission is denied"
                        % xml_path)
                return
            with open(xml_path, "r") as fr:
                tree = xmltodict.parse(fr.read())
            if tree["annotation"].get("object") is None:
                continue
            bboxes = tree["annotation"]["object"]
            if not isinstance(bboxes, list):
                bboxes = [bboxes]
            for bbx in bboxes:
                if bbx["name"] not in labels:
                    labels.append(bbx["name"])
            if len(labels) > 1 and self.series == "img":
                self.warning(
                    "More than one label in picture! labels:"
                    "%s picture: %s" % (labels, image_path))
                continue
            for temp_label in labels:
                if temp_label in self.get_labels_list_from_file():
                    label = temp_label
                else:
                    self.warning(
                        "Not a label %s in picture %s"
                        % (temp_label, image_path))
                if (root.prep_imagenet.get_label_from_txt_label
                        and label is None):
                    tmp_label = self.get_label_from_txt_label(temp_label)
                    if tmp_label in self.get_labels_list_from_file():
                        label = tmp_label
                    else:
                        self.warning(
                            "Not a label %s in picture %s"
                            % (tmp_label, image_path))
                result_labels.append(label)
        return result_labels

    def get_label_from_txt_label(self, temp_label):
        indices_to_categories = self.get_indices_to_categories_from_file()
        temp_label_list = temp_label.split("_")
        for word in temp_label_list:
            labels = []
            for key, category in indices_to_categories.items():
                if category.find(word) > -1:
                    if key not in labels:
                        labels.append(key)
            if len(labels) == 1:
                return labels[0]

    def get_indices_to_categories_from_file(self):
        indices_to_categories = {}
        with open(root.prep_imagenet.file_indices_to_categories, "r") as fin:
            for line in fin:
                index, category = line.split("\t")
                indices_to_categories[index] = category.strip()
        return indices_to_categories

    def get_labels_list_from_file(self):
        file_with_indices = root.prep_imagenet.file_with_indices
        with open(file_with_indices, 'r') as fin:
            return fin.read().splitlines()

    def get_labels(self, path):
        if root.prep_imagenet.get_label == "all_ways":
            labels = (
                self.get_label_from_image_name(path) or
                self.get_label_from_image_path(path) or
                self.get_label_from_xml(path))
        else:
            labels = getattr(
                self, "get_label_" + root.prep_imagenet.get_label)(path)
        if self.series == "DET":
            return [self.get_det_label_from_label(l) for l in labels]
        return labels

    def get_det_label_from_label(self, temp_label):
        path_hierarchy = root.prep_imagenet.file_hierarchy
        label = temp_label
        with open(path_hierarchy, "r") as file_hier:
            hierarchy = json.load(file_hier)
        for sub_label, com_label in hierarchy.items():
            if sub_label == temp_label:
                label = com_label
        return label

    def get_labels_int_txt(self, labels):
        for label in labels:
            if label not in self.labels_diff:
                self.labels_diff.add(label)
                self.label_ind += 1
                self.labels_int_txt[label] = self.label_ind

    def get_images_from_path(self, path, labels, set_type):
        self.labels = labels
        if not os.access(path, os.R_OK):
            self.warning(
                "File %s does not exist or read permission is denied" % path)
            return
        self.info(path)
        image_name = os.path.basename(path)
        if self.series == "img":
            if not self.labels:
                self.labels = self.get_labels(path)
            if not self.labels:
                self.k += 1
            else:
                self.get_labels_int_txt(self.labels)
        try:
            shape = Image.open(path).size
        except:
            self.warning(
                "Failed to determine the size of %s",
                path)
            return
        if self.series == "img":
            self.images[set_type][image_name] = {
                "path": path,
                "label": self.labels,
                "width": shape[0],
                "height": shape[1]}
        elif self.series == "DET":
            self.images[set_type][image_name] = {
                "path": path,
                "width": shape[0],
                "height": shape[1],
                "bbxs": []}
            self.fill_images_from_xml(path, set_type, image_name)
        else:
            raise ValueError("Unsupported series name: %s" % self.series)

    def fill_images_from_xml(self, image_path, set_type, image_name):
        (dir_images, dir_bboxes) = root.prep_imagenet.MAPPING[
            self.root_name][self.series][set_type]
        if image_path.find(dir_images) == -1:
            raise ValueError(
                "Image path %s is wrong. Should be in %s folder" %
                (image_path, dir_images))
        head, image_tail = image_path.split(dir_images)
        xml_path = head + dir_bboxes + image_tail.replace(".JPEG", ".xml")
        if not os.access(xml_path, os.R_OK):
            self.warning(
                "File %s does not exist or read permission is denied"
                % xml_path)
            return
        with open(xml_path, "r") as fr:
            tree = xmltodict.parse(fr.read())
        if tree["annotation"].get("object") is not None:
            bboxes = tree["annotation"]["object"]
            if not isinstance(bboxes, list):
                bboxes = [bboxes]
            for bbx in bboxes:
                bbx_lbl = bbx["name"]
                bbx_xmax = int(bbx["bndbox"]["xmax"])
                bbx_xmin = int(bbx["bndbox"]["xmin"])
                bbx_ymax = int(bbx["bndbox"]["ymax"])
                bbx_ymin = int(bbx["bndbox"]["ymin"])
                w = bbx_xmax - bbx_xmin
                h = bbx_ymax - bbx_ymin
                x = 0.5 * w + bbx_xmin
                y = 0.5 * h + bbx_ymin
                det_label = self.get_det_label_from_label(bbx_lbl)
                self.get_labels_int_txt((det_label,))
                dict_bbx = {"label": det_label,
                            "width": w,
                            "height": h,
                            "x": x,
                            "y": y}
                self.images[set_type][image_name]["bbxs"].append(dict_bbx)
        else:
            self.warning("Xml %s has no object attribute" % xml_path)

    def init_files_det(self):
        patgh_to_devkit = os.path.join(
            self.root_path, "ILSVRC2014_devkit/data/det_lists")
        if not os.path.exists(self.root_path):
            raise OSError("Path %s does not exist")
        if not os.path.exists(patgh_to_devkit):
            raise OSError("Path %s does not exist")
        self.get_images_from_files(
            os.path.join(patgh_to_devkit),
            self.root_path)
        self.save_images_to_json(self.images)

    def save_images_to_json(self, images):
        for set_type in (TEST, VALIDATION, TRAIN):
            fnme = os.path.join(
                self.root_path, IMAGES_JSON %
                (self.root_name, self.series, set_type))
            with open(fnme, 'w') as fp:
                json.dump(images[set_type], fp, indent=4)

    def get_images_from_files(self, folder, root_folder):
        for rt_path, _tmp, files in os.walk(folder, followlinks=True):
            for txt_file in files:
                self.info("Looking for images in %s:", txt_file)
                if os.path.splitext(txt_file)[1] == ".txt":
                    abs_path_text_file = os.path.join(rt_path, txt_file)
                    if txt_file.find("neg") == -1:
                        for set_type in (TEST, VALIDATION, TRAIN):
                            self.get_images_from_dict(
                                set_type, txt_file, abs_path_text_file,
                                root_folder)

    def get_images_from_dict(
            self, set_type, text_file, abs_path_text_files, root_folder):
        map_items = root.prep_imagenet.MAPPING[self.root_name][self.series]
        dir_set_type = {k: map_items[k][0] for k in (TEST, VALIDATION, TRAIN)}
        if text_file.find(set_type) != -1:
            images_from_file = self.get_dict_from_files(abs_path_text_files)
            for dict in images_from_file:
                tail = dict["tail_path"]
                labels = dict["labels"]
                path = os.path.join(os.path.join(
                    root_folder, dir_set_type[set_type]), tail)
                self.get_images_from_path(path, labels, set_type)

    def get_dict_from_files(self, abs_path_text_files):
        images_from_file = []
        labels = []
        with open(abs_path_text_files, "r") as fin:
            for line in fin:
                path_tail, _, label = line.partition(' ')
                path_tail = path_tail.strip() + ".JPEG"
                labels.append(label if label else None)
                images_from_file.append(
                    {"tail_path": path_tail, "labels": labels})
        return images_from_file

    def get_mean(self, set_type, image_name, image):
        for bbx in self.images[set_type][image_name]["bbxs"]:
            x = bbx["x"]
            y = bbx["y"]
            h_size = bbx["height"]
            w_size = bbx["width"]
            if h_size >= 8 and w_size >= 8 and h_size * w_size >= 256:
                self.sample_rect(image, x, y, h_size, w_size, None)

    def sample_rect(self, img, x_c, y_c, h_size, w_size, mean):
        nn_width = self.rect[0]
        nn_height = self.rect[1]
        x_min = x_c - w_size / 2
        y_min = y_c - h_size / 2
        x_max = x_min + w_size
        y_max = y_min + h_size
        img = img[y_min:y_max, x_min:x_max].copy()
        if img.shape[1] >= img.shape[0]:
            dst_width = nn_width
            dst_height = int(numpy.round(
                float(dst_width) * img.shape[0] / img.shape[1]))
        else:
            dst_height = nn_height
            dst_width = int(numpy.round(
                float(dst_height) * img.shape[1] / img.shape[0]))
        assert dst_width <= nn_width and dst_height <= nn_height
        img = cv2.resize(img, (dst_width, dst_height),
                         interpolation=cv2.INTER_LANCZOS4)
        dst_x_min = int(numpy.round(0.5 * (nn_width - dst_width)))
        dst_y_min = int(numpy.round(0.5 * (nn_height - dst_height)))
        dst_x_max = dst_x_min + img.shape[1]
        dst_y_max = dst_y_min + img.shape[0]
        assert dst_x_max <= nn_width and dst_y_max <= nn_height
        if mean is not None:
            sample = mean.copy()
            sample = sample.astype(numpy.uint8)
            sample[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = img
            return sample
        self.s_sum[dst_y_min:dst_y_max, dst_x_min:dst_x_max] += img
        self.s_count[dst_y_min:dst_y_max, dst_x_min:dst_x_max] += 1.0
        numpy.minimum(self.s_min[dst_y_min:dst_y_max, dst_x_min:dst_x_max],
                      img,
                      self.s_min[dst_y_min:dst_y_max, dst_x_min:dst_x_max])
        numpy.maximum(self.s_max[dst_y_min:dst_y_max, dst_x_min:dst_x_max],
                      img,
                      self.s_max[dst_y_min:dst_y_max, dst_x_min:dst_x_max])
        return None

    def get_dataset_img(
            self, set_type, image_name, image_path, image, mean, threshold):
        sample = self.transformation_image(image)
        self.get_original_labels_and_data(
            self.images[set_type][image_name]["label"],
            sample, set_type, image_path)
        self.s_sum += sample
        self.s_count += 1.0

    def get_dataset_DET(
            self, set_type, image_name, image_path, image, mean, threshold):
        for bbx in self.images[set_type][image_name]["bbxs"]:
            x = bbx["x"]
            y = bbx["y"]
            h_size = bbx["height"]
            w_size = bbx["width"]
            txt_label = bbx["label"]
            if (h_size >= threshold and w_size >= threshold and
                    h_size * w_size >= threshold * threshold):
                word_label = self.get_word_label_from_num(txt_label)
                int_label = self.labels_int_txt[txt_label]
                self.original_labels.append((word_label, int_label - 1))
                sample = self.prep_and_save_sample(
                    image, x, y, h_size, w_size, mean)
                sample.tofile(self.file_samples)
                self.count_samples[set_type] += 1

    def prep_and_save_sample(self, image, x, y, h, w, mean):
        sample = self.preprocess_sample(image)
        sample = self.sample_rect(sample, x, y, h, w, mean)
        return sample

    def preprocess_sample(self, data):
        if self._include_derivative:
            deriv_x = cv2.Sobel(
                cv2.cvtColor(data, cv2.COLOR_RGB2GRAY) if data.shape[-1] > 1
                else data, cv2.CV_32F, 1, 0, ksize=self._sobel_kernel_size)
            deriv_y = cv2.Sobel(
                cv2.cvtColor(data, cv2.COLOR_RGB2GRAY) if data.shape[-1] > 1
                else data, cv2.CV_32F, 0, 1, ksize=self._sobel_kernel_size)
            deriv = numpy.sqrt(numpy.square(deriv_x) + numpy.square(deriv_y))
            deriv -= deriv.min()
            mx = deriv.max()
            if mx:
                deriv *= 255.0 / mx
            deriv = numpy.clip(deriv, 0, 255).astype(numpy.uint8)
        if self.colorspace != "RGB" and not (data.shape[-1] == 1 and
                                             self.colorspace == "GRAY"):
            cv2.cvtColor(data, getattr(cv2, "COLOR_RGB2" + self.colorspace),
                         data)
        if self._include_derivative:
            shape = list(data.shape)
            shape[-1] += 1
            res = numpy.empty(shape, dtype=numpy.uint8)
            res[:, :, :-1] = data
            begindex = len(shape)
            res.ravel()[begindex::(begindex + 1)] = deriv.ravel()
        else:
            res = data
        assert res.dtype == numpy.uint8
        return res

    def save_dataset_to_file(self):
        original_data_dir = root.prep_imagenet.file_original_data
        original_labels_dir = root.prep_imagenet.file_original_labels
        count_samples_dir = root.prep_imagenet.file_count_samples
        matrix_file = root.prep_imagenet.file_matrix

        self.info(
            "Will remove old files :\n %s\n %s\n %s\n %s\n in 15 seconds. "
            "Make sure you want to continue" %
            (original_data_dir, original_labels_dir, count_samples_dir,
             matrix_file))

        time.sleep(15)

        if os.path.exists(original_data_dir):
            os.remove(original_data_dir)
        if os.path.exists(original_labels_dir):
            os.remove(original_labels_dir)
        if os.path.exists(count_samples_dir):
            os.remove(count_samples_dir)
        if os.path.exists(matrix_file):
            os.remove(matrix_file)

        diff_nums = []
        diff_words = []
        with open(self.path_to_categories, "r") as word_lab:
            for line in word_lab:
                num = line[:line.find("\t")]
                word = line[line.find("\t") + 1:line.find("\n")]
                if num not in diff_nums:
                    diff_nums.append(num)
                if word not in diff_words:
                    diff_words.append(word)
                if len(diff_nums) - len(diff_words) == 1:
                    word += "_%s" % num
                    diff_words.append(word)
                assert len(diff_nums) == len(diff_words)
                self.num_word.append((num, word))

        fnme = root.prep_imagenet.file_text_to_int_labels
        with open(fnme, 'r') as fp:
            self.labels_int_txt = json.load(fp)

        self.file_samples = open(original_data_dir, "wb")
        if self.series == "DET":
            set_type = TRAIN
            images_file_name = os.path.join(
                self.root_path,
                IMAGES_JSON % (self.root_name, self.series, set_type))
            try:
                self.info(
                    "Loading images info from %s to calculate mean" %
                    images_file_name)
                with open(images_file_name, 'r') as fp:
                    self.images[set_type] = json.load(fp)
            except Exception as e:
                self.exception("Failed to load %s", images_file_name)
                raise from_none(e)
            for image_name in sorted(self.images[set_type].keys()):
                image_fnme = self.images[set_type][image_name]["path"]
                image = self.decode_image(image_fnme)
                self.get_mean(set_type, image_name, image)
            mean, rdisp = self.transform_matrixes(self.s_sum, self.s_count)
        else:
            mean = None
        self.info("Mean image was calculated")
        for set_type in (TEST, VALIDATION, TRAIN):
            images_file_name = os.path.join(
                self.root_path,
                IMAGES_JSON % (self.root_name, self.series, set_type))
            try:
                self.info(
                    "Loading images info from %s to resize" %
                    images_file_name)
                with open(images_file_name, 'r') as fp:
                    self.images[set_type] = json.load(fp)
            except Exception as e:
                self.exception("Failed to load %s", images_file_name)
                raise from_none(e)
            for image_name in sorted(self.images[set_type].keys()):
                image_fnme = self.images[set_type][image_name]["path"]
                image = self.decode_image(image_fnme)
                getattr(self, "get_dataset_%s" % self.series)(
                    set_type, image_name, image_fnme, image, mean,
                    self.threshold_train)
        with open(original_labels_dir, "wb") as fout:
            self.info("Saving labels of images to %s" %
                      original_labels_dir)
            pickle.dump(self.original_labels, fout)
        with open(count_samples_dir, "w") as fout:
            logging.info("Saving count of test, validation and train to %s"
                         % count_samples_dir)
            json.dump(self.count_samples, fout)
        mean, rdisp = self.transform_matrixes(self.s_sum, self.s_count)
        self.save_matrixes(mean, rdisp, matrix_file)
        class_keys_path = os.path.join(
            self.root_path,
            "class_keys_%s_%s.json" % (self.root_name, self.series))
        self.info("Saving class_keys to %s" % class_keys_path)
        with open(class_keys_path, 'w') as fp:
            json.dump(self.class_keys, fp)
        self.file_samples.close()

    def get_word_label_from_num(self, label_num):
        label_word = None
        for num, word in self.num_word:
            if num == label_num:
                label_word = word
        return label_word

    def get_original_labels_and_data(self, txt_labels, sample, set_type, path):
        if self.series != "DET" and len(txt_labels) > 1:
            self.error("Too much labels for image")
        else:
            if not ((txt_labels is None or len(txt_labels) == 0)
                    and set_type == "test"):
                for txt_label in txt_labels:
                    word_label = self.get_word_label_from_num(txt_label)
                    int_label = self.labels_int_txt[txt_label]
                    self.original_labels.append((word_label, int_label - 1))
            else:
                self.original_labels.append((None, 0))
            sets = {"test": 0, "val": 1, "train": 2}
            self.class_keys[sets[set_type]].append(path)
            sample.tofile(self.file_samples)
            self.count_samples[set_type] += 1

    def transform_matrixes(self, s_sum, s_count):
        self.s_mean = s_sum / s_count
        mean = numpy.round(self.s_mean)
        numpy.clip(mean, 0, 255, mean)
        mean = mean.astype(opencl_types.dtypes[
            root.common.engine.precision_type])

        if self._include_derivative and self.series == "DET":
            mean = self.to_4ch(mean)
            mean[:, :, 3:4] = 0

        disp = self.s_max - self.s_min
        if self._include_derivative and self.series == "DET":
            disp = self.to_4ch(disp)

        rdisp = numpy.ones_like(disp.ravel())
        nz = numpy.nonzero(disp.ravel())
        rdisp[nz] = numpy.reciprocal(disp.ravel()[nz])
        rdisp.shape = disp.shape
        if self._include_derivative and self.series == "DET":
            rdisp[:, :, 3:4] = 1.0 / 128
        return mean, rdisp

    def to_4ch(self, array):
        assert len(array.shape) == 3
        aa = numpy.zeros(
            [array.shape[0], array.shape[1], 4], dtype=array.dtype)
        aa[:, :, 0:3] = array[:, :, 0:3]
        return aa

    def save_matrixes(self, mean, rdisp, matrix_file):
        out_path_mean = os.path.join(
            root.prep_imagenet.root_path, "mean_image.JPEG")
        scipy.misc.imsave(out_path_mean, self.s_mean)
        self.matrixes.append(mean)
        self.matrixes.append(rdisp)
        with open(matrix_file, "wb") as fout:
            self.info(
                "Saving mean, min and max matrix to %s" % matrix_file)
            pickle.dump(self.matrixes, fout)

    def transformation_image(self, image):
        if self.series == "DET":
            sample = image
        elif self.series == "img":
            sample = cv2.resize(
                image, self.rect, interpolation=cv2.INTER_LANCZOS4)
        else:
            raise ValueError("Unsupported series name: %s" % self.series)
        return sample

    def test_load_data(self):
        path_labels = os.path.join(
            root.prep_imagenet.root_path, "original_labels_%s_%s.pickle"
            % (root.prep_imagenet.root_name, root.prep_imagenet.series))

        path_data = os.path.join(
            root.prep_imagenet.root_path, "original_data_%s_%s.dat"
            % (root.prep_imagenet.root_name, root.prep_imagenet.series))
        rand = rnd.get()
        with open(path_labels, "rb") as fout:
            fout_file = pickle.load(fout)
        i = int(rand.rand() * len(fout_file))
        self.info("image number i %s" % i)
        label = fout_file[i]
        self.info("label %s" % str(label))
        self.file_samples = open(path_data, "rb")
        if self.series == "DET":
            sample = numpy.zeros([216, 216, 4], dtype=numpy.uint8)
        if self.series == "img":
            sample = numpy.zeros([256, 256, 3], dtype=numpy.uint8)
        self.file_samples.seek(i * sample.nbytes)
        self.file_samples.readinto(sample)
        plt.imshow(sample[:, :, 0:3].copy(), interpolation="nearest")
        plt.show()

    def save_validation_to_forward(self):
        set_type = "validation"
        self.info("Resized validation to query of Neural Network")

        original_labels_dir = os.path.join(
            self.root_path, "original_labels_%s_%s_forward.pickle" %
            (self.root_name, self.series))
        path_for_matrixes = os.path.join(
            self.root_path, "matrixes_%s_%s.pickle" %
            (self.root_name, self.series))
        original_data_dir = os.path.join(
            self.root_path, "original_data_%s_%s_forward.dat" %
            (self.root_name, self.series))
        with open(path_for_matrixes, "rb") as file_matr:
            matrixes = pickle.load(file_matr)
        mean = matrixes[0]
        fnme = os.path.join(self.root_path,
                            IMAGES_JSON %
                            (self.root_name, self.series, set_type))
        try:
            self.info("Loading images info from %s to resize" % fnme)
            with open(fnme, 'r') as fp:
                self.images[set_type] = json.load(fp)
        except OSError:
            self.exception("Failed to load %s", fnme)
        self.file_samples = open(original_data_dir, "wb")
        for image_name, _val in sorted(self.images[set_type].items()):
            image_fnme = self.images[set_type][image_name]["path"]
            image = self.decode_image(image_fnme)
            getattr(self, "get_dataset_%s" % self.series)(
                set_type, image_name, image, mean, self.threshold_val)

        self.info("Saving images to %s" % original_data_dir)
        with open(original_labels_dir, "wb") as fout:
            self.info("Saving labels of images to %s" %
                      original_labels_dir)
            pickle.dump(self.original_labels, fout)
        self.file_samples.close()

    def init_dataset(self):
        if self.series == "DET":
            self.init_files_det()
        elif self.series == "img":
            self.init_files_img()
        else:
            self.error("Unknow series. Please choose DET or img")
        if len(self.labels_int_txt) != root.prep_imagenet.classes_count:
            self.warning(
                "Wrong number of classes - %s" % len(self.labels_int_txt))
        else:
            fnme = root.prep_imagenet.file_text_to_int_labels
            with open(fnme, 'w') as fp:
                json.dump(self.labels_int_txt, fp)

    def run(self):
        self.initialize()
        getattr(self, root.prep_imagenet.command_to_run)()
        self.info("End of job")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(PreparationImagenet().run())
