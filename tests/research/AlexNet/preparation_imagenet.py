#!/usr/bin/python3
# encoding: utf-8
"""
Created on Dec 10, 2014

This script to work with Imagenet dataset.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import cv2
import json
import logging
import numpy
import pickle
from PIL import Image
import os
import scipy.misc
import sys

from veles.compat import from_none
from veles.config import root
import veles.opencl_types as opencl_types
from veles.znicz.external import xmltodict
from veles.znicz.tests.research.imagenet.processor import Processor


IMAGES_JSON = "images_imagenet_%s_%s_%s.json"
TEST = "test"
TRAIN = "train"
VALIDATION = "val"

root.prep_imagenet.root_name = "imagenet"
root.prep_imagenet.series = "img"
root.prep_imagenet.root_path = os.path.join(
    root.common.test_dataset_root, "AlexNet", root.prep_imagenet.root_name)

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
    # "save_dataset_to_file" "init_dataset"
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


class Main(Processor):
    def __init__(self, **kwargs):
        super(Main, self).__init__(**kwargs)
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
        self.labels_diff = []
        self.original_labels = []
        self.command_to_run = root.prep_imagenet.command_to_run
        self.count_samples = {TEST: 0, TRAIN: 0, VALIDATION: 0}
        self.matrixes = []

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
                self.labels_diff.append(label)
                self.label_ind += 1
                self.labels_int_txt[label] = self.label_ind

    def get_images_from_path(self, path, labels, set_type):
        self.labels = labels
        if not os.access(path, os.R_OK):
            self.warning(
                "File %s does not exist or read permission is denied" % path)
            return
        image_name = os.path.basename(path)
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
        self.images[set_type][image_name] = {
            "path": path,
            "label": self.labels,
            "width": shape[0],
            "height": shape[1]}

    def init_files_det(self):
        self.get_images_from_files(
            os.path.join(
                self.root_path, "ILSVRC2014_devkit/data/det_lists"),
            self.root_path)
        self.save_images_to_json(self.images)
        self.info("Label is None! %s times" % self.k)

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

    def save_dataset_to_file(self):
        original_data_dir = root.prep_imagenet.file_original_data
        original_labels_dir = root.prep_imagenet.file_original_labels
        count_samples_dir = root.prep_imagenet.file_count_samples
        self.file_samples = open(original_data_dir, "wb")
        for set_type in (TEST, VALIDATION, TRAIN):
            images_file_name = os.path.join(
                self.root_path,
                IMAGES_JSON % (self.root_name, self.series, set_type))
            try:
                self.info(
                    "Loading images info from %s to resize" % images_file_name)
                with open(images_file_name, 'r') as fp:
                    self.images[set_type] = json.load(fp)
            except Exception as e:
                self.exception("Failed to load %s", images_file_name)
                raise from_none(e)
            for image_name in sorted(self.images[set_type].keys()):
                image_fnme = self.images[set_type][image_name]["path"]
                image = self.decode_image(image_fnme)
                sample = self.transformation_image(image)
                self.get_original_labels_and_data(
                    self.images[set_type][image_name]["label"],
                    sample, set_type)
                self.s_sum += sample
                self.s_count += 1.0
        with open(original_labels_dir, "wb") as fout:
            self.info("Saving labels of images to %s" %
                      original_labels_dir)
            pickle.dump(self.original_labels, fout)
        with open(count_samples_dir, "w") as fout:
            logging.info("Saving count of test, validation and train to %s"
                         % count_samples_dir)
            json.dump(self.count_samples, fout)
        self.transform_and_save_matrixes(self.s_sum, self.s_count)
        self.file_samples.close()

    def get_original_labels_and_data(self, txt_labels, sample, set_type):
        text_to_int_labels_dir = root.prep_imagenet.file_text_to_int_labels
        with open(text_to_int_labels_dir, "r") as fin:
            labels_int_txt = json.load(fin)
        int_labels = [labels_int_txt[txt_label] for txt_label in txt_labels]
        if self.series != "DET" and len(int_labels) > 1:
            self.error("Too mach labels for image")
        else:
            for int_label in int_labels:
                sample.tofile(self.file_samples)
                self.original_labels.append(int_label)
                self.count_samples[set_type] += 1

    def transform_and_save_matrixes(self, s_sum, s_count):
        matrix_file = root.prep_imagenet.file_matrix
        self.s_mean = s_sum / s_count
        mean = numpy.round(self.s_mean)
        numpy.clip(mean, 0, 255, mean)
        mean = mean.astype(opencl_types.dtypes[root.common.precision_type])
        disp = self.s_max - self.s_min
        rdisp = numpy.ones_like(disp.ravel())
        nz = numpy.nonzero(disp.ravel())
        rdisp[nz] = numpy.reciprocal(disp.ravel()[nz])
        rdisp.shape = disp.shape
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
        sample = cv2.resize(image, self.rect, interpolation=cv2.INTER_LANCZOS4)
        return sample

    def init_dataset(self):
        if self.series == "DET":
            self.init_files_det()
        elif self.series == "img":
            self.init_files_img()
        else:
            self.error("Unknow series. Please choose DET or img")
        if len(self.labels_int_txt) != root.prep_imagenet.classes_count:
            self.error(
                "Wrong number of classes - %s"
                % len(self.labels_int_txt))
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
    sys.exit(Main().run())
