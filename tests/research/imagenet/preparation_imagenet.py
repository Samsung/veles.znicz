#!/usr/bin/python3
# encoding: utf-8
"""
Created on Jun 26, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
'''
This script to work with Imagenet dataset.

.. argparse::
   :module: veles.znicz.tests.research.imagenet.preparation_imagenet
   :func: create_args_parser_sphinx
   :prog: preparation_imagenet

'''


try:
    import argcomplete
except:
    pass
import argparse
try:
    import cv2
except ImportError:
    import warnings
    warnings.warn("Failed to import OpenCV bindings")
import json
import logging
import numpy
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
import sys

import veles.config as config
import veles.random_generator as rnd
from veles.znicz.external import xmltodict
from veles.znicz.tests.research.imagenet.processor import Processor


IMAGENET_BASE_PATH = os.path.join(config.root.common.test_dataset_root,
                                  "imagenet")
IMAGES_JSON = "images_imagenet_%s_%s_%s_%s.json"
# year, series, set_type, stage


class Main(Processor):
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    LOG_LEVEL_MAP = {"debug": logging.DEBUG, "info": logging.INFO,
                     "warning": logging.WARNING, "error": logging.ERROR}

    def __init__(self, **kwargs):
        self.imagenet_dir_path = None
        self.year = None
        self.series = None
        self.fnme = None
        self.stage = None
        self.count_classes = 0
        self.count_dirs = 40
        self.matrixes = []
        self.images_json = {
            "test": {},  # dict: {"path", "label", "bbx": [{bbx}, {bbx}, ...]}
            "validation": {},
            "train": {}
            }
        self.names_labels = {
            "test": [],
            "validation": [],
            "train": []
            }
        self.ind_labels = []
        self.do_save_resized_images = kwargs.get("do_save_resized_images",
                                                 False)
        self.rect = kwargs.get("rect", (256, 256))
        self._sobel_kernel_size = kwargs.get(
            "sobel_kernel_size",
            config.get(config.root.imagenet.sobel_ksize) or 5)
        self._colorspace = kwargs.get(
            "colorspace", config.get(config.root.imagenet.colorspace) or "RGB")
        if self._colorspace == "GRAY":
            self._crop_color = self._crop_color[0]
        self._include_derivative = kwargs.get(
            "derivative", config.get(config.root.imagenet.derivative) or True)
        super(Main, self).__init__(**kwargs)
        self.s_mean = numpy.zeros(self.rect + (3,), dtype=numpy.float64)
        self.s_count = numpy.zeros_like(self.s_mean)
        self.s_min = numpy.empty_like(self.s_mean)
        self.s_min[:] = 255
        self.s_max = numpy.zeros_like(self.s_mean)

    @property
    def colorspace(self):
        return self._colorspace

    @staticmethod
    def init_parser():
        """
        Creates the command line argument parser.
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", type=str, default="info",
                            choices=Main.LOG_LEVEL_MAP.keys(),
                            help="set logging verbosity level")
        parser.add_argument("-y", "--year", type=str, default="temp",
                            choices=["2014", "2013", "temp"],
                            help="set dataset year")
        parser.add_argument("-s", "--series", type=str, default="img",
                            choices=["img", "DET"],
                            help="set dataset type")
        parser.add_argument("-st", "--stage", default=0,
                            help="set stage")
        parser.add_argument("command_to_run", type=str, default="",
                            choices=["draw_bbox", "resize", "init",
                                     "split_valid", "split_dataset",
                                     "generate_negative", "init_split_dataset",
                                     "negative_split_dataset",
                                     "resize_split_dataset"],
                            help="run functions:"
                                 " 'draw_bbox' run function which generate"
                                 " image with bboxes, 'resize' run function"
                                 " which resized images to bboxes, 'init' run"
                                 " function which generate json file"
                                 " 'split_valid' split validation images"
                                 " to dirs - classes. 'split_dataset' create"
                                 " new dataset: split imagenet dataset to"
                                 " specified number of folders - count_dirs"
                                 " 'generate_negative' generate negative"
                                 " images in the dataset. 'init_split_datas"
                                 " et' - run init for all dirs in split"
                                 " dataset. 'negative_split_dataset' run"
                                 " generate_negative or all dirs in split"
                                 " dataset. 'resize_split_dataset' run resize"
                                 " for all dirs in split dataset")
        try:
            class NoEscapeCompleter(argcomplete.CompletionFinder):
                def quote_completions(self, completions, *args, **kwargs):
                    return completions
            NoEscapeCompleter()(parser)  # pylint: disable=E1102
        except:
            pass
        return parser

    def init_files(self, path_imagenet):
        self.images_iter = {
            "train": {},  # dict: {"path", "label", "bbx": [{bbx}, {bbx}, ...]}
            "validation": {},
            "test": {}
            }
        MAPPING = {"% s" % self.year:
                  {"img":
                   {"train":
                    ("ILSVRC2012_img_train", "ILSVRC2012_bbox_train_v2"),
                    "validation":
                    ("ILSVRC2012_img_val", "ILSVRC2012_bbox_val_v2"),
                    "test": ("ILSVRC2012_img_test", "")},
                   "DET":
                   {"train":
                    ("ILSVRC2014_DET_train", "ILSVRC2014_DET_bbox_train"),
                    "validation":
                    ("ILSVRC2013_DET_val", "ILSVRC2013_DET_bbox_val"),
                    "test": ("ILSVRC2013_DET_test", "")}}}
        self.imagenet_dir_path = path_imagenet
        self.info("Looking for images in %s:", self.imagenet_dir_path)
        int_labels_dir = os.path.join(self.imagenet_dir_path,
                                      "labels_int_%s_%s_%s.txt" %
                                      (self.year, self.series, self.stage))
        # finding dirs for images and bboxes
        zero_write = True
        map_items = MAPPING[self.year][self.series].items()
        ind = 1
        for set_type, (dir_images, dir_bboxes) in sorted(map_items):
            print("------", set_type, dir_images, dir_bboxes)
            path = os.path.join(self.imagenet_dir_path, dir_images)
            self.info("Scanning JPG %s...", path)
            temp_images = self.images_iter[set_type]
            for root_path, _tmp, files in os.walk(path, followlinks=True):
                # print("ROOT=", root)
                for f in files:
                    if os.path.splitext(f)[1] == ".JPEG":
                        f_path = os.path.join(root_path, f)
                        #--------------------------------------------
                        # KGG check if dirs have duplicates filenames
                        # KGG it was checked - no diplicates; code commented
                        # temp_image = temp_images.get(f)
                        # if temp_image != None:
                        #    self.error("Duplicate file name:\n"
                        #               "present: %s\n    found: %s",
                        #               temp_image[0], f_path)
                        # -------------------------------------------

                        # lets find label from the filename
                        label = None
                        temp_label = f.split('_')[0]
                        if (temp_label[0] == 'n'):
                            label = temp_label
                        bbx = []
                        temp_images[f] = {"path": f_path, "label": label,
                                          "bbxs": bbx}
                    else:
                        self.warning("Unexpected file in dir %s", f)
            if dir_bboxes != "":
                path = os.path.join(self.imagenet_dir_path, dir_bboxes)
                self.info("Scanning xml %s...", path)
                for root_path, _tmp, files in os.walk(path, followlinks=True):
                    for f in files:
                        if os.path.splitext(f)[1] == ".xml":
                            image_fname = os.path.splitext(f)[0] + ".JPEG"
                            xml_path = os.path.join(root_path, f)
                            with open(xml_path, "r") as fr:
                                tree = xmltodict.parse(fr.read())
                            temp_bbx = tree["annotation"]["object"]
                            if type(temp_bbx) is not list:
                                temp_bbx = [temp_bbx]
                            for bbx in temp_bbx:
                                bbx_lbl = bbx["name"]
                                bbx_xmax = int(bbx["bndbox"]["xmax"])
                                bbx_xmin = int(bbx["bndbox"]["xmin"])
                                bbx_ymax = int(bbx["bndbox"]["ymax"])
                                bbx_ymin = int(bbx["bndbox"]["ymin"])
                                bbx_ang = 0  # degree
                                # we left 0 for purpose
                                # we will check for zerro
                                # and will not use rotation
                                w = bbx_xmax - bbx_xmin
                                h = bbx_ymax - bbx_ymin
                                x = 0.5 * w + bbx_xmin
                                y = 0.5 * h + bbx_ymin
                                image_lbl = self.images_iter[
                                    set_type][image_fname]["label"]
                                if (bbx_lbl == image_lbl or
                                    (bbx_lbl != image_lbl and
                                     image_lbl is None and
                                     bbx_lbl is not None)):
                                    label = bbx_lbl
                                elif (bbx_lbl != image_lbl and
                                      image_lbl is not None):
                                    label = image_lbl
                                else:
                                    label = None
                                    self.warning(
                                        "could not find image"
                                        "label in file %s",
                                        self.images_iter[
                                            set_type][image_fname]["path"])
                                dict_bbx = {"label": label,
                                            "angle": bbx_ang,
                                            "width": w,
                                            "height": h,
                                            "x": x,
                                            "y": y}
                                self.images_iter[set_type][
                                    image_fname]["bbxs"].append(dict_bbx)
                                found = False
                                if zero_write:
                                    file_ind_labels = open(int_labels_dir, "w")
                                    file_ind_labels.write(
                                        "0\tnegative image\n%s\t%s\n"
                                        % (ind, label))
                                    file_ind_labels.close()
                                    ind += 1
                                    zero_write = False
                                if label in open(int_labels_dir).read():
                                    found = True
                                file_ind_labels.close()
                                file_ind_labels = open(int_labels_dir, "a")
                                if found is False:
                                    line_int_label = "%s\t%s\n" % (ind, label)
                                    file_ind_labels.write(line_int_label)
                                    ind += 1
                                file_ind_labels.close()
            try:
                os.mkdir(self.imagenet_dir_path)
            except OSError:
                pass
            fnme = os.path.join(
                self.imagenet_dir_path, IMAGES_JSON %
                (self.year, self.series, set_type, self.stage))
            # image - dict: "path_to_img", "label", "bbx": [{bbx}, {bbx}, ...]
            with open(fnme, 'w') as fp:
                json.dump(self.images_iter[set_type], fp)

        return None

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
            cv2.cvtColor(data, getattr(cv2, "COLOR_RGB2" + self._colorspace),
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
        return (res, deriv)

    def to_4ch(self, a):
        assert len(a.shape) == 3
        aa = numpy.zeros([a.shape[0], a.shape[1], 4], dtype=a.dtype)
        aa[:, :, 0:3] = a[:, :, 0:3]
        return aa

    def generate_resized_dataset(self, path):
        self.info("Resized dataset")
        original_labels = []
        int_word_labels = []
        zero_train = True
        self.imagenet_dir_path = path
        original_labels_dir = os.path.join(
            self.imagenet_dir_path, "original_labels_%s_%s_%s.pickle" %
            (self.year, self.series, self.stage))
        matrix_dir = os.path.join(
            self.imagenet_dir_path, "matrixes_%s_%s_%s.pickle" %
            (self.year, self.series, self.stage))
        count_samples_dir = os.path.join(
            self.imagenet_dir_path, "count_samples_%s_%s_%s.json" %
            (self.year, self.series, self.stage))
        labels_int_dir = os.path.join(
            self.imagenet_dir_path, "labels_int_%s_%s_%s.txt" %
            (self.year, self.series, self.stage))
        original_data_dir = os.path.join(
            self.imagenet_dir_path, "original_data_%s_%s_%s.dat" %
            (self.year, self.series, self.stage))
        file_labels_int = open(labels_int_dir, "r")
        for line in file_labels_int:
            int_label = line[:line.find("\t")]
            word_label = line[line.find("\t") + 1:line.find("\n")]
            int_word_labels.append((int_label, word_label))
            self.count_classes += 1
        set_type = "train"
        fnme = os.path.join(self.imagenet_dir_path,
                            IMAGES_JSON %
                            (self.year, self.series, set_type, self.stage))
        try:
            self.info("Loading images info from %s to calculate mean image"
                      % fnme)
            with open(fnme, 'r') as fp:
                self.images_json[set_type] = json.load(fp)
        except:
            self.exception("Failed to load %s", fnme)
        for f, _val in sorted(self.images_json[set_type].items()):
            image_fnme = self.images_json[set_type][f]["path"]
            image = self.decode_image(image_fnme)
            path_to_save = image_fnme[:image_fnme.rfind("/")]
            path_to_save = path_to_save[:path_to_save.rfind("/")]
            path_to_save = os.path.join(path_to_save, "n00000000")
            if f.find("negative_image") != -1:
                self.info("f %s " % f)
                self.sample_rect(
                    image, image.shape[1] / 2, image.shape[0] / 2,
                    image.shape[0], image.shape[1], 0, None)
            for bbx in self.images_json[set_type][f]["bbxs"]:
                x = bbx["x"]
                y = bbx["y"]
                h_size = bbx["height"]
                w_size = bbx["width"]
                ang = bbx["angle"]
                self.sample_rect(image, x, y, h_size, w_size, ang, None)
        self.s_mean /= self.s_count

        # Convert mean to 0..255 uint8
        mean = numpy.round(self.s_mean)
        numpy.clip(mean, 0, 255, mean)
        mean = self.to_4ch(mean).astype(numpy.uint8)
        mean[:, :, 3:4] = 128

        # Calculate reciprocal dispersion
        disp = self.to_4ch(self.s_max - self.s_min)
        rdisp = numpy.ones_like(disp.ravel())
        nz = numpy.nonzero(disp.ravel())
        rdisp[nz] = numpy.reciprocal(disp.ravel()[nz])
        rdisp.shape = disp.shape
        rdisp[:, :, 3:4] = 1.0 / 128

        self.info("Mean image is calculated")
        out_path_mean = os.path.join(path_to_save,
                                     "mean_image_%s.JPEG" % self.year)
        scipy.misc.imsave(out_path_mean, self.s_mean)
        self.matrixes.append(mean)
        self.matrixes.append(rdisp)
        test_count = 0
        validation_count = 0
        train_count = 0
        sample_count = 0
        labels_count = 0
        self.f_samples = open(original_data_dir, "wb")
        for set_type in ("test", "validation", "train"):
            fnme = os.path.join(
                self.imagenet_dir_path, IMAGES_JSON %
                (self.year, self.series, set_type, self.stage))
            try:
                self.info("Loading images info from %s to resize" % fnme)
                with open(fnme, 'r') as fp:
                    self.images_json[set_type] = json.load(fp)
            except:
                self.exception("Failed to load %s", fnme)
            for f, _val in sorted(self.images_json[set_type].items()):
                image_fnme = self.images_json[set_type][f]["path"]
                image = self.decode_image(image_fnme)
                i = 0
                if f.find("negative_image") != -1:
                    if set_type == "test":
                        test_count += 1
                    elif set_type == "validation":
                        validation_count += 1
                    elif set_type == "train":
                        train_count += 1
                    else:
                        self.error("Wrong set type")
                    self.prep_and_save_sample(
                        image, f, image.shape[1] / 2, image.shape[0] / 2,
                        image.shape[0], image.shape[1], 0, mean)
                    sample_count += 1
                    original_labels.append(0)
                    labels_count += 1
                for bbx in self.images_json[set_type][f]["bbxs"]:
                    self.info("*****Resized image %s *****" %
                              self.images_json[set_type][f]["path"])
                    if set_type == "test":
                        test_count += 1
                    elif set_type == "validation":
                        validation_count += 1
                    elif set_type == "train":
                        train_count += 1
                        if zero_train:
                            mean.tofile(self.f_samples)
                            original_labels.append(0)
                            train_count += 1
                            sample_count += 1
                            labels_count += 1
                            self.count_classes += 1
                            zero_train = False
                    else:
                        self.error("Wrong set type")
                    x = bbx["x"]
                    y = bbx["y"]
                    h_size = bbx["height"]
                    w_size = bbx["width"]
                    label = bbx["label"]
                    ang = bbx["angle"]
                    name = f[:f.rfind(".")] + ("_%s_bbx.JPEG" % i)
                    self.prep_and_save_sample(image, name, x, y, h_size,
                                              w_size, ang, mean)
                    sample_count += 1
                    for (int_label, word_label) in int_word_labels:
                        if label == word_label:
                            original_labels.append(int_label)
                            labels_count += 1
                    self.count_samples = [test_count, validation_count,
                                          train_count]
                    i += 1
            self.info("Saving images to %s" % original_data_dir)
        with open(original_labels_dir, "wb") as fout:
            self.info("Saving labels of images to %s" %
                      original_labels_dir)
            pickle.dump(original_labels, fout)
        with open(matrix_dir, "wb") as fout:
            self.info("Saving mean, min and max matrix to %s" % matrix_dir)
            pickle.dump(self.matrixes, fout)
        with open(count_samples_dir, "w") as fout:
            self.info("Saving count of test, validation and train to %s"
                      % count_samples_dir)
            json.dump(self.count_samples, fout)
        self.info("labels_count %s sample_count %s"
                  % (labels_count, sample_count))
        assert labels_count == sample_count
        assert sample_count == train_count + validation_count + test_count
        self.f_samples.close()

    def prep_and_save_sample(self, image, name, x, y, h, w, ang, mean):
        out_dir = os.path.join(config.root.common.cache_dir,
                               "tmp_imagenet")
        sample = self.sample_rect(image, x, y, h, w, ang, mean[:, :, 0:3])
        sample_sobel = self.preprocess_sample(sample)
        sobel = sample_sobel[1]
        sample = sample_sobel[0]
        sample.tofile(self.f_samples)
        image_to_save = sample[:, :, 0:3]
        if self.do_save_resized_images:
            out_path_sample = os.path.join(
                out_dir, "all_samples/%s" % name)
            scipy.misc.imsave(out_path_sample, image_to_save)
            out_path_sobel = os.path.join(
                out_dir, "all_samples/sobel_%s" % name)
            scipy.misc.imsave(out_path_sobel, sobel)

    def sample_rect(self, img, x_c, y_c, h_size, w_size, ang, mean):
        nn_width = self.rect[0]
        nn_height = self.rect[1]
        rot_matrix = cv2.getRotationMatrix2D((x_c, y_c), 360 - ang, 1)
        img_rotate = cv2.warpAffine(img, rot_matrix,
                                    (img.shape[1], img.shape[0]))
        x_min = x_c - w_size / 2
        y_min = y_c - h_size / 2
        x_max = x_min + w_size
        y_max = y_min + h_size
        img = img_rotate[y_min:y_max, x_min:x_max].copy()
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
                         interpolation=cv2.INTER_AREA)
        dst_x_min = int(numpy.round(0.5 * (nn_width - dst_width)))
        dst_y_min = int(numpy.round(0.5 * (nn_height - dst_height)))
        dst_x_max = dst_x_min + img.shape[1]
        dst_y_max = dst_y_min + img.shape[0]
        assert dst_x_max <= nn_width and dst_y_max <= nn_height
        if mean is not None:
            sample = mean.copy()
            sample[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = img
            return sample
        self.s_mean[dst_y_min:dst_y_max, dst_x_min:dst_x_max] += img
        self.s_count[dst_y_min:dst_y_max, dst_x_min:dst_x_max] += 1.0
        numpy.minimum(self.s_min[dst_y_min:dst_y_max, dst_x_min:dst_x_max],
                      img,
                      self.s_min[dst_y_min:dst_y_max, dst_x_min:dst_x_max])
        numpy.maximum(self.s_max[dst_y_min:dst_y_max, dst_x_min:dst_x_max],
                      img,
                      self.s_max[dst_y_min:dst_y_max, dst_x_min:dst_x_max])
        return None

    def split_validation_to_dirs(self, path):
        self.imagenet_dir_path = path
        path_to_img_validation = os.path.join(
            self.imagenet_dir_path, "ILSVRC2012_img_val")
        path_to_bbox_validation = os.path.join(
            self.imagenet_dir_path, "ILSVRC2012_bbox_val_v2")
        set_type = "validation"
        fnme = os.path.join(self.imagenet_dir_path, IMAGES_JSON %
                            (self.year, self.series, set_type, self.stage))
        try:
            self.info("Loading images info from %s"
                      " to split validation to dirs" % fnme)
            with open(fnme, 'r') as fp:
                self.images_json[set_type] = json.load(fp)
        except:
            self.exception("Failed to load %s", fnme)
        for image_name, _val in sorted(self.images_json[set_type].items()):
            image_path = self.images_json[set_type][image_name]["path"]
            bbox_name = image_name.replace(".JPEG", ".xml")
            bbox_path = os.path.join(path_to_bbox_validation, bbox_name)
            zero_move = True
            for bbx in self.images_json[set_type][image_name]["bbxs"]:
                label = bbx["label"]
                path_img_valid_dir = os.path.join(path_to_img_validation,
                                                  str(label))
                path_xml_valid_dir = os.path.join(path_to_bbox_validation,
                                                  str(label))
                try:
                    os.mkdir(path_img_valid_dir)
                except:
                    pass
                try:
                    os.mkdir(path_xml_valid_dir)
                except:
                    pass
                if zero_move:
                    try:
                        os.rename(image_path, os.path.join(path_img_valid_dir,
                                                           image_name))
                    except:
                        pass
                    try:
                        os.rename(bbox_path, os.path.join(path_xml_valid_dir,
                                                          bbox_name))
                    except:
                        pass
                    zero_move = False

    def split_dataset(self, count_dirs, path):
        self.imagenet_dir_path = path
        self.common_split_dir = os.path.join(
            IMAGENET_BASE_PATH, "%s_split_%s" % (self.year, self.stage))
        try:
            os.mkdir(self.common_split_dir, 0o775)
        except:
            pass
        labels_int_dir = os.path.join(
            self.imagenet_dir_path, "labels_int_%s_%s_%s.txt" %
            (self.year, self.series, self.stage))
        file_labels_int = open(labels_int_dir, "r")
        self.classes = []
        for line in file_labels_int:
            label_class = line[line.find("\t") + 1:line.find("\n")]
            self.classes.append(label_class)
            self.count_classes += 1
        self.classes.sort()
        self.count_classes -= 1  # - one negative class - 0
        if self.count_classes:
            for i_patch in range(1, count_dirs + 1):
                path_to_patch_folder = os.path.join(
                    self.common_split_dir, "%s" % (i_patch))
                path_to_train_bbox_folder = os.path.join(
                    path_to_patch_folder, "ILSVRC2012_bbox_train_v2")
                path_to_train_img_folder = os.path.join(
                    path_to_patch_folder, "ILSVRC2012_img_train")
                path_to_val_bbox_folder = os.path.join(
                    path_to_patch_folder, "ILSVRC2012_bbox_val_v2")
                path_to_val_img_folder = os.path.join(
                    path_to_patch_folder, "ILSVRC2012_img_val")
                try:
                    os.mkdir(path_to_patch_folder, 0o775)
                except:
                    pass
                try:
                    os.mkdir(path_to_train_bbox_folder, 0o775)
                except:
                    pass
                try:
                    os.mkdir(path_to_train_img_folder, 0o775)
                except:
                    pass
                try:
                    os.mkdir(path_to_val_bbox_folder, 0o775)
                except:
                    pass
                try:
                    os.mkdir(path_to_val_img_folder, 0o775)
                except:
                    pass
                count_classes_in_dir = int(self.count_classes / count_dirs)
                for i in range((i_patch - 1) * count_classes_in_dir,
                               i_patch * count_classes_in_dir):
                    path_train_img_from = os.path.join(os.path.join(
                        self.imagenet_dir_path,
                        "ILSVRC2012_img_train"), self.classes[i])
                    path_train_img_to = os.path.join(
                        path_to_train_img_folder, self.classes[i])
                    try:
                        os.symlink(path_train_img_from, path_train_img_to)
                    except:
                        pass
                    path_train_bbox_from = os.path.join(os.path.join(
                        self.imagenet_dir_path,
                        "ILSVRC2012_bbox_train_v2"), self.classes[i])
                    path_train_bbox_to = os.path.join(
                        path_to_train_bbox_folder, self.classes[i])
                    try:
                        os.symlink(path_train_bbox_from, path_train_bbox_to)
                    except:
                        pass
                    path_valid_img_from = os.path.join(os.path.join(
                        self.imagenet_dir_path,
                        "ILSVRC2012_img_val"), self.classes[i])
                    path_valid_img_to = os.path.join(
                        path_to_val_img_folder, self.classes[i])
                    try:
                        os.symlink(path_valid_img_from, path_valid_img_to)
                    except:
                        pass
                    path_valid_bbox_from = os.path.join(os.path.join(
                        self.imagenet_dir_path,
                        "ILSVRC2012_bbox_val_v2"), self.classes[i])
                    path_valid_bbox_to = os.path.join(
                        path_to_val_bbox_folder, self.classes[i])
                    try:
                        os.symlink(path_valid_bbox_from, path_valid_bbox_to)
                    except:
                        pass

    def init_split_dataset(self, count_dirs):
        self.common_split_dir = os.path.join(
            IMAGENET_BASE_PATH, "%s_split_%s" % (self.year, self.stage))
        for i_patch in range(1, count_dirs + 1):
            self.year = str(i_patch)
            path_to_patch_folder = os.path.join(
                self.common_split_dir, "%s" % (i_patch))
            self.info("run init_files in %s" % path_to_patch_folder)
            self.init_files(path_to_patch_folder)

    def resize_split_dataset(self, count_dirs):
        self.common_split_dir = os.path.join(
            IMAGENET_BASE_PATH, "%s_split_%s" % (self.year, self.stage))
        for i_patch in range(1, count_dirs + 1):
            self.year = str(i_patch)
            path_to_patch_folder = os.path.join(
                self.common_split_dir, "%s" % (i_patch))
            self.info("run generate_resized_dataset in %s"
                      % path_to_patch_folder)
            self.generate_resized_dataset(path_to_patch_folder)

    def negative_split_dataset(self, count_dirs):
        self.common_split_dir = os.path.join(
            IMAGENET_BASE_PATH, "%s_split_%s" % (self.year, self.stage))
        for i_patch in range(1, count_dirs + 1):
            self.year = str(i_patch)
            path_to_patch_folder = os.path.join(
                self.common_split_dir, "%s" % (i_patch))
            self.info("run generate_negative_images in %s"
                      % path_to_patch_folder)
            self.generate_negative_images(path_to_patch_folder)

    def generate_negative_images(self, path):
        self.imagenet_dir_path = path
        min_size_max_side = 128
        max_count_negative_in_class = 100
        count_negative_in_class = 0
        class_is_new = False
        prev_label = ""
        rand = rnd.get()
        for set_type in ("test", "validation", "train"):
            fnme = os.path.join(
                self.imagenet_dir_path, IMAGES_JSON %
                (self.year, self.series, set_type, self.stage))
            try:
                with open(fnme, 'r') as fp:
                    self.images_json[set_type] = json.load(fp)
            except:
                self.exception("Failed to load %s", fnme)
            ind = 0
            for f, _val in sorted(self.images_json[set_type].items()):
                image_fnme = self.images_json[set_type][f]["path"]
                path_to_save = image_fnme[:image_fnme.rfind("/")]
                path_to_save = path_to_save[:path_to_save.rfind("/")]
                path_to_save = os.path.join(path_to_save, "n00000000")
                try:
                    os.mkdir(path_to_save)
                except:
                    pass
                image = self.decode_image(image_fnme)
                if len(self.images_json[set_type][f]["bbxs"]) == 1:
                    bbx = self.images_json[set_type][f]["bbxs"][0]
                    x = bbx["x"]
                    y = bbx["y"]
                    h_size = bbx["height"]
                    w_size = bbx["width"]
                    ang = bbx["angle"]
                    current_label = bbx["label"]
                    if current_label != prev_label:
                        class_is_new = True
                    else:
                        class_is_new = False
                    prev_label = current_label
                    x_min = x - w_size / 2
                    y_min = y - h_size / 2
                    x_max = x_min + w_size
                    y_max = y_min + h_size
                    if ang > 0:
                        Matr = cv2.getRotationMatrix2D((x, y), 360 - ang, 1)
                        image = cv2.warpAffine(
                            image, Matr, (image.shape[1], image.shape[0]))
                    if w_size >= h_size:
                        min_size_min_side = (h_size * min_size_max_side
                                             / w_size)
                    else:
                        min_size_min_side = (w_size * min_size_max_side
                                             / h_size)
                    if min_size_min_side < 64:
                        min_size_min_side = 64
                    if class_is_new is True:
                        count_negative_in_class = 0
                    if count_negative_in_class < max_count_negative_in_class:
                        for _ in range(16):
                            stripe = rand.randint(4)
                            if stripe == 0:
                                x_neg = x_min / 2
                                w_neg = x_min
                                h_neg = w_neg * h_size / w_size
                                if (w_neg < min_size_min_side or
                                        h_neg < min_size_min_side or w_neg
                                        > image.shape[1] or
                                        h_neg > image.shape[0]):
                                    continue
                                y_neg = h_neg / 2 + (
                                    image.shape[0] - h_neg) * rand.rand()
                                sample_neg = self.resize(
                                    image, x_neg, y_neg, h_neg, w_neg)
                                path_to_save_neg = os.path.join(
                                    path_to_save,
                                    "negative_image_%s_%s" % (ind, f))
                                ind += 1
                                count_negative_in_class += 1
                                scipy.misc.imsave(path_to_save_neg, sample_neg)
                                break
                            if stripe == 1:
                                y_neg = y_min / 2
                                h_neg = y_min
                                w_neg = h_neg * w_size / h_size
                                if (w_neg < min_size_min_side or
                                        h_neg < min_size_min_side or w_neg
                                        > image.shape[1] or
                                        h_neg > image.shape[0]):
                                    continue
                                x_neg = w_neg / 2 + (
                                    image.shape[1] - w_neg) * rand.rand()
                                sample_neg = self.resize(
                                    image, x_neg, y_neg, h_neg, w_neg)
                                path_to_save_neg = os.path.join(
                                    path_to_save,
                                    "negative_image_%s_%s" % (ind, f))
                                ind += 1
                                count_negative_in_class += 1
                                scipy.misc.imsave(path_to_save_neg, sample_neg)
                                break
                            if stripe == 2:
                                x_neg = (image.shape[1] - x_max) / 2 + x_max
                                w_neg = image.shape[1] - x_max
                                h_neg = w_neg * h_size / w_size
                                if (w_neg < min_size_min_side or
                                        h_neg < min_size_min_side or w_neg
                                        > image.shape[1] or
                                        h_neg > image.shape[0]):
                                    continue
                                y_neg = h_neg / 2 + (
                                    image.shape[0] - h_neg) * rand.rand()
                                sample_neg = self.resize(
                                    image, x_neg, y_neg, h_neg, w_neg)
                                path_to_save_neg = os.path.join(
                                    path_to_save,
                                    "negative_image_%s_%s" % (ind, f))
                                ind += 1
                                count_negative_in_class += 1
                                scipy.misc.imsave(path_to_save_neg, sample_neg)
                                break
                            if stripe == 3:
                                y_neg = (image.shape[0] - y_max) / 2 + y_max
                                h_neg = image.shape[0] - y_max
                                w_neg = h_neg * w_size / h_size
                                if (w_neg < min_size_min_side or
                                        h_neg < min_size_min_side or w_neg
                                        > image.shape[1] or
                                        h_neg > image.shape[0]):
                                    continue
                                x_neg = w_neg / 2 + (
                                    image.shape[1] - w_neg) * rand.rand()
                                sample_neg = self.resize(
                                    image, x_neg, y_neg, h_neg, w_neg)
                                path_to_save_neg = os.path.join(
                                    path_to_save,
                                    "negative_image_%s_%s" % (ind, f))
                                ind += 1
                                count_negative_in_class += 1
                                scipy.misc.imsave(path_to_save_neg, sample_neg)
                                break

    def resize(self, img, x, y, h, w):
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x_min + w
        y_max = y_min + h
        sample_neg = img[y_min:y_max, x_min:x_max]
        return sample_neg

    def generate_images_with_bbx(self, path):
        """
        self.imagenet_dir_path = "%s/%s" % (IMAGENET_BASE_PATH, self.year)
        try:
            shutil.copytree(self.imagenet_dir_path,
                            os.path.join(self.imagenet_dir_path,
                                         "images_with_bb %s" % self.year))
        except:
            print("Failed to copy")
            pass
        """
        fontsize = 25
        label_txt = ""
        digits_word = []
        font = ImageFont.truetype(
            os.path.join(config.root.common.test_dataset_root,
                         "arial.ttf"), fontsize)
        cached_data_fnme = path
        categories_path = os.path.join(cached_data_fnme,
                                       "categories_imagenet_%s_list.txt"
                                       % self.year)
        self.categories = open(categories_path, "r")
        for line in self.categories:
            digits_label = line[:line.find("\t")]
            word_label = line[line.find("\t") + 1:line.find("\n")]
            digits_word.append((digits_label, word_label))
        self.categories.close()
        for set_type in ("test", "validation", "train"):
            fnme = os.path.join(
                cached_data_fnme, IMAGES_JSON %
                (self.year, self.series, set_type, self.stage))
            try:
                with open(fnme, 'r') as fp:
                    self.images_json[set_type] = json.load(fp)
            except:
                self.exception("Failed to load %s", fnme)
            for f, _val in sorted(self.images_json[set_type].items()):
                image_path = Image.open(self.images_json[set_type][f]["path"])
                draw = ImageDraw.Draw(image_path)
                for bbx in self.images_json[set_type][f]["bbxs"]:
                    x = bbx["x"]
                    y = bbx["y"]
                    h = bbx["height"]
                    w = bbx["width"]
                    label = bbx["label"]
                    x_min = x - w / 2
                    y_min = y - h / 2
                    x_max = x_min + w
                    y_max = y_min + h
                    self.info("*****draw bbx in image %s *****" %
                              self.images_json[set_type][f]["path"])
                    for dig_word in digits_word:
                        if dig_word[0] == label:
                            label_txt = dig_word[1]
                    draw.text((x_min + 5, y_min), label_txt, fill=255,
                              font=font)
                    draw.line((x_min, y_min, x_min, y_max),
                              fill="red", width=3)
                    draw.line((x_min, y_min, x_max, y_min),
                              fill="red", width=3)
                    draw.line((x_min, y_max, x_max, y_max),
                              fill="red", width=3)
                    draw.line((x_max, y_min, x_max, y_max),
                              fill="red", width=3)
                path_to_image = self.images_json[set_type][f]["path"]
                ind_path = path_to_image.rfind("/")
                try:
                    os.mkdir(path_to_image[:ind_path])
                except OSError:
                    pass
                path_to_image = path_to_image.replace("temp",
                                                      "images_with_bb")
                image_path.save(path_to_image, "JPEG")

    def run(self):
        """Image net utility
        """
        parser = Main.init_parser()
        args = parser.parse_args()
        self.setup(level=Main.LOG_LEVEL_MAP[args.verbose])
        self.year = args.year
        self.series = args.series
        self.command_to_run = args.command_to_run
        self.stage = args.stage
        if self.command_to_run == "init":
            self.init_files(os.path.join(IMAGENET_BASE_PATH, self.year))
        elif self.command_to_run == "draw_bbox":
            self.generate_images_with_bbx(os.path.join(IMAGENET_BASE_PATH,
                                                       self.year))
        elif self.command_to_run == "resize":
            self.generate_resized_dataset(os.path.join(IMAGENET_BASE_PATH,
                                                       self.year))
        elif self.command_to_run == "split_valid":
            self.split_validation_to_dirs(os.path.join(IMAGENET_BASE_PATH,
                                                       self.year))
        elif self.command_to_run == "split_dataset":
            self.split_dataset(self.count_dirs,
                               os.path.join(IMAGENET_BASE_PATH, self.year))
        elif self.command_to_run == "resize_split_dataset":
            self.resize_split_dataset(self.count_dirs)
        elif self.command_to_run == "init_split_dataset":
            self.init_split_dataset(self.count_dirs)
        elif self.command_to_run == "generate_negative":
            self.generate_negative_images(os.path.join(IMAGENET_BASE_PATH,
                                                       self.year))
        elif self.command_to_run == "negative_split_dataset":
            self.negative_split_dataset(self.count_dirs)
        else:
            self.info("Choose command to run: 'all' run all functions,"
                      "'draw_bbox' run function which generate"
                      "image with bboxes, 'resize' run function"
                      "which resized images to bboxes, 'init' run"
                      " function which generate json file")

        self.info("End of job")
        return Main.EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(Main().run())
