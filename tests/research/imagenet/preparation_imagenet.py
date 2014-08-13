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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
import shutil
import sys
import veles.config as config
import veles.formats as formats
import veles.prng as rnd
import veles.znicz.conv as conv
from veles.znicz.external import xmltodict
from veles.znicz.tests.research.imagenet.processor import Processor
import veles.znicz.tests.research.imagenet.background_detection as back_det

IMAGENET_BASE_PATH = os.path.join(config.root.common.test_dataset_root,
                                  "imagenet/temp")

#IMAGENET_BASE_PATH = "/data/veles/datasets/imagenet/2014"
IMAGES_JSON = "images_imagenet_%s_%s_%s_%s.json"
# year, series, set_type, stage

INDICES_HIERARCHY_FILE = os.path.join(IMAGENET_BASE_PATH,
                                      "2014/indices_hierarchy.txt")


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
        self.background = None
        self.count_classes = 0
        self.count_dirs = 40
        self.do_bboxes_map = False
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
                                                 True)
        self.rect = kwargs.get("rect", (216, 216))
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
        self.s_mean = numpy.zeros((768, 1024) + (3,), dtype=numpy.float64)
        self.s_count = numpy.ones_like(self.s_mean)
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
                            help="set dataset year")
        parser.add_argument("-s", "--series", type=str, default="img",
                            choices=["img", "DET"],
                            help="set dataset type")
        parser.add_argument("-st", "--stage", default=0,
                            help="set stage")
        parser.add_argument("-w", "--snapshot", type=str, default="",
                            help="snapshot path")
        parser.add_argument("-b", "--background", type=str, default="mean",
                            choices=["mean", "mean_and_image",
                                     "expanding_blur", "blur",
                                     "random_last_line"],
                            help="set backgroud of resized images")
        parser.add_argument("command_to_run", type=str, default="",
                            choices=["draw_bbox", "resize", "init",
                                     "split_valid", "split_dataset",
                                     "generate_negative", "init_split_dataset",
                                     "negative_split_dataset",
                                     "resize_split_dataset", "split_train",
                                     "test_segmentation", "min_max_shape",
                                     "generate_negative_DET", "test_load",
                                     "remove_back", "visualize",
                                     "remove_back_split_dataset",
                                     "resize_validation", "none_bboxes"],
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
                    ("ILSVRC2012_img_val", "ILSVRC2012_bbox_val_v3"),
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
        path_hierarchy = os.path.join(IMAGENET_BASE_PATH,
                                      "2014/hierarchy_2014_DET_train_0.json")
        if self.series == "DET":
            with open(path_hierarchy, "r") as file_hier:
                hierarchy = json.load(file_hier)
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
                # print("ROOT=", root_path)
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
                        try:
                            shape = Image.open(f_path).size
                        except:
                            shape = (-1, -1)
                            self.warning("Failed to determine the size of %s",
                                         f_path)
                        temp_images[f] = {"path": f_path, "label": label,
                                          "bbxs": bbx, "width": shape[0],
                                          "height": shape[1]}
                    else:
                        self.warning("Unexpected file in dir %s", f)
            if dir_bboxes != "":
                path = os.path.join(self.imagenet_dir_path, dir_bboxes)
                self.info("Scanning xml %s...", path)
                for root_path, _tmp, files in os.walk(path, followlinks=True):
                    # print("ROOT=", root_path)
                    for f in files:
                        if os.path.splitext(f)[1] == ".xml":
                            image_fname = os.path.splitext(f)[0] + ".JPEG"
                            xml_path = os.path.join(root_path, f)
                            with open(xml_path, "r") as fr:
                                tree = xmltodict.parse(fr.read())
                            if tree["annotation"].get("object") is not None:
                                temp_bbx = tree["annotation"]["object"]
                                if type(temp_bbx) is not list:
                                    temp_bbx = [temp_bbx]
                                for bbx in temp_bbx:
                                    bbx_lbl = bbx["name"]
                                    bbx_xmax = int(bbx["bndbox"]["xmax"])
                                    bbx_xmin = int(bbx["bndbox"]["xmin"])
                                    bbx_ymax = int(bbx["bndbox"]["ymax"])
                                    bbx_ymin = int(bbx["bndbox"]["ymin"])
                                    if self.stage == 0:
                                        bbx_ang = 0.0  # degree
                                    # we left 0 for purpose
                                    # we will check for zerro
                                    # and will not use rotation
                                    w = bbx_xmax - bbx_xmin
                                    h = bbx_ymax - bbx_ymin
                                    x = 0.5 * w + bbx_xmin
                                    y = 0.5 * h + bbx_ymin
                                    image_lbl = self.images_iter[
                                        set_type][image_fname]["label"]
                                    if self.series == "img":
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
                                                    set_type][
                                                    image_fname]["path"])
                                    if self.series == "DET":
                                        for sub_label, com_label in sorted(
                                                hierarchy.items()):
                                            if sub_label == bbx_lbl:
                                                label = com_label
                                    self.info("label %s" % label)
                                    dict_bbx = {"label": label,
                                                "conf": 1.0,
                                                "angle": bbx_ang,
                                                "width": w,
                                                "height": h,
                                                "x": x,
                                                "y": y}
                                    self.images_iter[set_type][
                                        image_fname]["bbxs"].append(dict_bbx)
                                    found = False
                                    if zero_write:
                                        file_ind_labels = open(int_labels_dir,
                                                               "w")
                                        file_ind_labels.write(
                                            "0\tnegative_image\n%s\t%s\n"
                                            % (ind, label))
                                        file_ind_labels.close()
                                        ind += 1
                                        zero_write = False
                                    if label in open(int_labels_dir).read():
                                        found = True
                                    file_ind_labels.close()
                                    file_ind_labels = open(int_labels_dir, "a")
                                    if found is False:
                                        line_int_label = "%s\t%s\n" % (ind,
                                                                       label)
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
                json.dump(self.images_iter[set_type], fp, indent=4)

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
        return res

    def to_4ch(self, a):
        assert len(a.shape) == 3
        aa = numpy.zeros([a.shape[0], a.shape[1], 4], dtype=a.dtype)
        aa[:, :, 0:3] = a[:, :, 0:3]
        return aa

    def visualize_snapshot(self, path, limit):
        with open(path, "rb") as snap_file:
            snapshot = pickle.load(snap_file)
        for forward in snapshot.fwds:
            weights = forward.weights.mem
            if weights is not None and isinstance(forward, conv.Conv):
                size = weights.shape[1]
                weights = weights.reshape(weights.shape[0],
                                          weights.size // weights.shape[0])
                pics = []
                for _kernel_number in range(weights.shape[0]):
                    n_channels = int(size / (forward.kx * forward.ky))
                    mem = weights[_kernel_number].ravel()[:size]
                    kernel = mem.reshape([forward.ky, forward.kx, n_channels])
                    for ch in range(n_channels):
                        pics.append(
                            formats.norm_image(
                                kernel[:, :, ch:ch
                                       + 1].reshape(forward.ky, forward.kx)))
                        if len(pics) >= limit:
                            break
                    if len(pics) >= limit:
                        break
                self.info("pics is ready")
                self.info("pics %s" % len(pics))
                figure = plt.figure()
                figure.clf()
                if len(pics) > 0:
                    n_cols = formats.roundup(
                        int(numpy.round(numpy.sqrt(len(pics)))), 4)
                    n_rows = int(numpy.ceil(len(pics) / n_cols))
                self.info("n_cols %s" % n_cols)
                self.info("n_rows %s" % n_rows)
                i = 0
                for _row in range(n_rows):
                    for _col in range(n_cols):
                        ax = figure.add_subplot(n_rows, n_cols, i + 1)
                        ax.cla()
                        ax.axis('off')
                        ax.imshow(pics[i],
                                  interpolation=("nearest", "lanczos")[1],
                                  cmap=cm.Greys_r)
                        i += 1
                        if i >= len(pics):
                            break
                    if i >= len(pics):
                        break
                self.info("show")
                plt.show()

    def resize_validation(self, path):
        _display = os.getenv("DISPLAY")
        if _display is not None:
            os.unsetenv("DISPLAY")
        self.imagenet_dir_path = path
        #self.year = "forward"
        set_type = "validation"
        original_labels = []
        int_word_labels = []
        self.info("Resized validation to query of Neural Network")

        original_labels_dir = os.path.join(
            self.imagenet_dir_path, "original_labels_%s_%s_%s_forward.pickle" %
            (self.year, self.series, self.stage))
        path_for_matrixes = os.path.join(
            self.imagenet_dir_path, "matrixes_%s_%s_%s.pickle" %
            (self.year, self.series, self.stage))
        original_data_dir = os.path.join(
            self.imagenet_dir_path, "original_data_%s_%s_%s_forward.dat" %
            (self.year, self.series, self.stage))
        labels_int_dir = os.path.join(
            self.imagenet_dir_path, "labels_int_%s_%s_%s.txt" %
            (self.year, self.series, self.stage))
        try:
            file_labels_int = open(labels_int_dir, "r")
            for line in file_labels_int:
                int_label = line[:line.find("\t")]
                word_label = line[line.find("\t") + 1:line.find("\n")]
                int_word_labels.append((int_label, word_label))
        except:
            pass
        with open(path_for_matrixes, "rb") as file_matr:
            matrixes = pickle.load(file_matr)
        mean = matrixes[0]
        fnme = os.path.join(self.imagenet_dir_path,
                            "images_imagenet_%s_%s_%s_%s_forward.json" %
                            (self.year, self.series, set_type, self.stage))
        try:
            self.info("Loading images info from %s to resize" % fnme)
            with open(fnme, 'r') as fp:
                self.images_json[set_type] = json.load(fp)
        except:
            self.exception("Failed to load %s", fnme)
        sample_count = 0
        labels_count = 0
        self.f_samples = open(original_data_dir, "wb")
        for f, _val in sorted(self.images_json[set_type].items()):
            image_fnme = self.images_json[set_type][f]["path"]
            image = self.decode_image(image_fnme)
            i = 0
            for bbx in self.images_json[set_type][f]["bbxs"]:
                self.info("*****Resized image %s *****" %
                          self.images_json[set_type][f]["path"])
                x = bbx["x"]
                y = bbx["y"]
                h_size = bbx["height"]
                w_size = bbx["width"]
                label = bbx["label"]
                self.info("label %s" % label)
                ang = bbx["angle"]
                name = f[:f.rfind(".")] + ("_%s_bbx.JPEG" % i)
                if h_size >= 1 and w_size >= 1 and h_size * w_size >= 1:
                    self.prep_and_save_sample(image, name, x, y, h_size,
                                              w_size, ang, mean)
                    sample_count += 1
                    if self.series == "DET":
                        imagenet_dir = os.path.join(IMAGENET_BASE_PATH,
                                                    "2014")
                        classes_word_path = os.path.join(
                            imagenet_dir,
                            "classes_200_2014_DET_train_0.json")
                        with open(classes_word_path, 'r') as fp:
                            int_word_labels = json.load(fp)
                    if len(int_word_labels):
                        for (int_label, word_label) in int_word_labels:
                            if label == word_label:
                                original_labels.append(int_label)
                                labels_count += 1
                    else:
                        original_labels.append(0)
                        labels_count += 1
                    i += 1
        self.info("Saving images to %s" % original_data_dir)
        with open(original_labels_dir, "wb") as fout:
            self.info("Saving labels of images to %s" %
                      original_labels_dir)
            pickle.dump(original_labels, fout)
        self.info("labels_count %s sample_count %s"
                  % (labels_count, sample_count))
        assert labels_count == sample_count
        self.f_samples.close()

    def calculate_bbox_is_none(self, path):
        self.imagenet_dir_path = path
        file_none_bboxes = os.path.join(
            self.imagenet_dir_path, "bbox_is_none.txt")
        set_type = "train"
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
            if (len(self.images_json[set_type][f]["bbxs"]) == 0
                    and f.find("negative_image") == -1):
                self.info("Image without bboxes %s" % image_fnme)
                with open(file_none_bboxes, "a") as fin:
                    fin.write("%s\n" % image_fnme)

    def generate_resized_dataset(self, path):
        _display = os.getenv("DISPLAY")
        if _display is not None:
            os.unsetenv("DISPLAY")

        self.info("Resized dataset")
        original_labels = []
        int_word_labels = []
        zero_train = True
        self.do_negative = False
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
        set_type = "validation"
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
                self.do_negative = True
                self.sample_rect(
                    image, image.shape[1] / 2, image.shape[0] / 2,
                    image.shape[0], image.shape[1], 0, None)
            h_scale = 768 / image.shape[0]
            w_scale = 1024 / image.shape[1]
            for bbx in self.images_json[set_type][f]["bbxs"]:
                x = bbx["x"]
                y = bbx["y"]
                h_size = bbx["height"]
                w_size = bbx["width"]
                ang = bbx["angle"]
                if self.do_bboxes_map:
                    x *= w_scale
                    w_size *= w_scale
                    y *= h_scale
                    h_size *= h_scale
                    self.s_mean[
                        (y - h_size // 2):(y - h_size // 2 + h_size),
                        (x - w_size // 2):(x - w_size // 2 + w_size)] += (255,
                                                                          255,
                                                                          255)
                    continue
                else:
                    if h_size >= 8 and w_size >= 8 and h_size * w_size >= 256:
                        self.sample_rect(image, x, y, h_size, w_size, ang,
                                         None)
        if self.do_bboxes_map:
            self.s_mean /= numpy.max(self.s_mean)
            self.s_mean *= 255
        else:
            self.s_mean /= self.s_count

        # Convert mean to 0..255 uint8
        mean = numpy.round(self.s_mean)
        numpy.clip(mean, 0, 255, mean)
        mean = self.to_4ch(mean).astype(numpy.uint8)
        mean[:, :, 3:4] = 0

        # Calculate reciprocal dispersion
        disp = self.to_4ch(self.s_max - self.s_min)
        rdisp = numpy.ones_like(disp.ravel())
        nz = numpy.nonzero(disp.ravel())
        rdisp[nz] = numpy.reciprocal(disp.ravel()[nz])
        rdisp.shape = disp.shape
        rdisp[:, :, 3:4] = 1.0 / 128

        self.info("Mean image is calculated")
        if self.do_negative is True and self.series == "img":
            out_path_mean = os.path.join(path_to_save,
                                         "mean_image_%s.JPEG" % self.year)
        else:
            out_path_mean = os.path.join(self.imagenet_dir_path,
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
                    x = bbx["x"]
                    y = bbx["y"]
                    h_size = bbx["height"]
                    w_size = bbx["width"]
                    label = bbx["label"]
                    self.info("label %s" % label)
                    ang = bbx["angle"]
                    name = f[:f.rfind(".")] + ("_%s_bbx.JPEG" % i)
                    if h_size >= 8 and w_size >= 8 and h_size * w_size >= 256:
                        if set_type == "test":
                            test_count += 1
                        elif set_type == "validation":
                            validation_count += 1
                        elif set_type == "train":
                            train_count += 1
                            if zero_train:
                                for i in range(0, 20):
                                    mean.tofile(self.f_samples)
                                    original_labels.append(0)
                                    train_count += 1
                                    sample_count += 1
                                    labels_count += 1
                                self.count_classes += 1
                                zero_train = False
                        else:
                            self.error("Wrong set type")

                        self.prep_and_save_sample(image, name, x, y, h_size,
                                                  w_size, ang, mean)
                        sample_count += 1
                        if self.series == "DET":
                            imagenet_dir = os.path.join(IMAGENET_BASE_PATH,
                                                        "2014")
                            classes_word_path = os.path.join(
                                imagenet_dir,
                                "classes_200_2014_DET_train_0.json")
                            with open(classes_word_path, 'r') as fp:
                                int_word_labels = json.load(fp)
                        for (int_label, word_label) in int_word_labels:
                            if label == word_label:
                                original_labels.append(int_label)
                                labels_count += 1
                        i += 1
                self.count_samples = [test_count, validation_count,
                                      train_count]
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

        if _display is not None:
            os.putenv("DISPLAY", _display)

    def prep_and_save_sample(self, image, name, x, y, h, w, ang, mean):
        out_dir = os.path.join(config.root.common.cache_dir,
                               "tmp_imagenet")
        sample = self.preprocess_sample(image)
        sample = self.sample_rect(sample, x, y, h, w, ang, mean)
        sample.tofile(self.f_samples)
        image_to_save = sample[:, :, 0:3]
        image_to_save_sobel = sample[:, :, 3:4].reshape(sample.shape[0],
                                                        sample.shape[1])
        if self.do_save_resized_images:
            out_path_sample = os.path.join(
                out_dir, "all_samples/%s" % name)
            scipy.misc.imsave(out_path_sample, image_to_save)
            out_path_sobel = os.path.join(
                out_dir, "all_samples/sobel_%s" % name)
            scipy.misc.imsave(out_path_sobel, image_to_save_sobel)

    def test_load_data(self, path):
        self.imagenet_dir = path
        path_labels = os.path.join(self.imagenet_dir,
                                   "original_labels_%s_%s_%s.pickle"
                                   % (self.year, self.series, self.stage))

        path_data = os.path.join(self.imagenet_dir,
                                 "original_data_%s_%s_%s.dat"
                                 % (self.year, self.series, self.stage))
        rand = rnd.get()
        with open(path_labels, "rb") as fout:
            fout_file = pickle.load(fout)
        i = int(rand.rand() * len(fout_file))
        self.info("image number i %s" % i)
        label = fout_file[i]
        path_to_ind_labels = os.path.join(
            self.imagenet_dir, "labels_int_%s_%s_%s.txt"
            % (self.year, self.series, self.stage))
        self.info("label %s" % label)
        if label == 0:
            label_num = "n00000000"
            label_word = "negative_image"
        if self.series == "img":
            labels_ind = []
            with open(path_to_ind_labels, "r") as ind_lab:
                for line in ind_lab:
                    ind = line[:line.find("\t")]
                    lab = line[line.find("\t") + 1:line.find("\n")]
                    labels_ind.append((ind, lab))
            for ind, lab in labels_ind:
                if label == ind:
                    label_num = lab
            self.info("label num %s" % label_num)
            path_to_categories = os.path.join(IMAGENET_BASE_PATH,
                                              "2014/indices_to_categories.txt")
            num_word = []
            with open(path_to_categories, "r") as word_lab:
                for line in word_lab:
                    num = line[:line.find("\t")]
                    word = line[line.find("\t") + 1:line.find("\n")]
                    num_word.append((num, word))
            for num, word in num_word:
                if num == label_num:
                    label_word = word
            self.info("categories %s" % label_word)
        if self.series == "DET":
            path_to_labels_word = os.path.join(
                IMAGENET_BASE_PATH,
                "2014/classes_200_2014_DET_train_0.json")
            with open(path_to_labels_word, "r") as label_word:
                label_cat = json.load(label_word)
                for (ind, label_w) in label_cat:
                    if label == ind:
                        label_num = label_w
            self.info("label num %s" % label_num)
            path_to_categories = os.path.join(IMAGENET_BASE_PATH,
                                              "2014/indices_to_categories.txt")
            num_word = []
            with open(path_to_categories, "r") as word_lab:
                for line in word_lab:
                    num = line[:line.find("\t")]
                    word = line[line.find("\t") + 1:line.find("\n")]
                    num_word.append((num, word))
            for num, word in num_word:
                if num == label_num:
                    label_word = word
            self.info("categories %s" % label_word)
        self.file_samples = open(path_data, "rb")
        sample = numpy.zeros([216, 216, 4], dtype=numpy.uint8)
        self.file_samples.seek(i * sample.nbytes)
        self.file_samples.readinto(sample)
        plt.imshow(sample[:, :, 0:3].copy(), interpolation="nearest")
        plt.show()

    def get_validation(self):
        list_image = "/home/lpodoynitsina/Desktop/bird.txt"
        # list_image: grep label_train *.xml > path_to_file.txt
        # label_train: n01440764 (example)
        # path_to_file.txt: "/home/lpodoynitsina/Desktop/fish.txt"
        file_image = open(list_image, "r")
        names_jpeg = []
        names_xml = []
        for line in file_image:
            name_xml = line[:line.find(".xml") + 4]
            name_jpeg = name_xml.replace(".xml", ".JPEG")
            names_xml.append(name_xml)
            names_jpeg.append(name_jpeg)
        for name_image in names_jpeg:
            path_to_copy = os.path.join(IMAGENET_BASE_PATH,
                                        ("temp/ILSVRC2012_img_val/%s"
                                         % name_image))
            path_from_copy = os.path.join(IMAGENET_BASE_PATH,
                                          ("2014/ILSVRC2012_img_val/%s"
                                           % name_image))
            shutil.copyfile(path_from_copy, path_to_copy)
        for name_bbx in names_xml:
            path_to_copy = os.path.join(IMAGENET_BASE_PATH,
                                        ("temp/ILSVRC2012_bbox_val_v2/%s"
                                         % name_bbx))
            path_from_copy = os.path.join(IMAGENET_BASE_PATH,
                                          ("2014/ILSVRC2012_bbox_val_v2/%s"
                                           % name_bbx))
            shutil.copyfile(path_from_copy, path_to_copy)

    def rotate_test(self):
        image_fnme = "/home/lpodoynitsina/Desktop/1.JPEG"
        img = self.decode_image(image_fnme)
        ang = 30
        mean_path = "/home/lpodoynitsina/Desktop/mean_image.JPEG"
        mean = self.decode_image(mean_path)
        xmin = 1
        ymin = 198
        xmax = 317
        ymax = 352
        w_size = xmax - xmin
        h_size = ymax - ymin
        x_c = 0.5 * w_size + xmin
        y_c = 0.5 * h_size + ymin
        sample = self.sample_rect(img, x_c, y_c, h_size, w_size, ang, mean)
        image_to_save = sample[:, :, 0:3]
        out_path_sample = os.path.join("/home/lpodoynitsina/Desktop/out.JPEG")
        scipy.misc.imsave(out_path_sample, image_to_save)

    def sample_rect(self, img, x_c, y_c, h_size, w_size, ang, mean):
        aperture = self.rect[0]
        rot_matrix = cv2.getRotationMatrix2D((x_c, y_c), 360 - ang, 1)
        img_rotate = cv2.warpAffine(img, rot_matrix,
                                    (img.shape[1], img.shape[0]))
        if self.background != "mean":
            max_size = max(w_size, h_size)
            x_min = max(x_c - max_size / 2, 0)
            y_min = max(y_c - max_size / 2, 0)
            x_max = min(x_min + max_size, img.shape[1])
            y_max = min(y_min + max_size, img.shape[0])
            img = img_rotate[y_min:y_max, x_min:x_max]
            scale = aperture / max_size
            dst_bbox = tuple([int(numpy.round(img.shape[i] * scale))
                              for i in (1, 0)])
            img = cv2.resize(img, dst_bbox, interpolation=cv2.INTER_LANCZOS4)
            if not (img.shape[0] == img.shape[1] == aperture):
                if img.shape[0] < aperture:
                    dst_x_min = 0
                    dst_y_min = (aperture - img.shape[0]
                                 if y_c - max_size / 2 < 0 else 0)
                    if self.background != "mean_and_image":
                        line = (img[0:1, 0:img.shape[1], :3]
                                if y_c - max_size / 2 < 0 else
                                img[(img.shape[0] - 1):img.shape[0],
                                    0:img.shape[1], :3])
                        background = numpy.zeros([dst_y_min, aperture, 3],
                                                 dtype=numpy.uint8)
                        new_line = line.copy()
                        for i in range(background.shape[0] - 1, -1, -1):
                            if self.background != "random_last_line":
                                new_line = line.copy()
                                new_line = new_line.reshape(line.shape[1], 3)
                                numpy.random.shuffle(new_line)
                                new_line = new_line.reshape(line.shape[0],
                                                            line.shape[1], 3)
                                background[i] = new_line[:]
                            elif self.background != "blur":
                                blur = cv2.blur(new_line, ksize=(3, 3))
                                background[i] = blur[:]
                                new_line = blur
                            elif self.background != "expanding_blur":
                                blur = cv2.blur(new_line, ksize=(i + 1, i + 1))
                                background[i] = blur[:]
                                new_line = blur
                            else:
                                self.error("Wrong background")
                elif img.shape[1] < aperture:
                    dst_y_min = 0
                    dst_x_min = (aperture - img.shape[1]
                                 if x_c - max_size / 2 < 0 else 0)
                    if self.background != "mean_and_image":
                        line = (img[0:img.shape[0], 0:1, :3]
                                if x_c - max_size / 2 < 0 else
                                img[0:img.shape[0],
                                    (img.shape[1] - 1):img.shape[1], :3])
                        background = numpy.zeros([aperture, dst_x_min, 3],
                                                 dtype=numpy.uint8)
                        # TO_DO: added different backgrounds for this case
                else:
                    assert False
            else:
                dst_x_min = 0
                dst_y_min = 0

            dst_x_max = dst_x_min + img.shape[1]
            dst_y_max = dst_y_min + img.shape[0]
            # TO_DO: added background to the image
        else:
            nn_width = self.rect[0]
            nn_height = self.rect[1]
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
                             interpolation=cv2.INTER_LANCZOS4)
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
        if self.series == "img":
            path_to_img_validation = os.path.join(
                self.imagenet_dir_path, "ILSVRC2012_img_val")
            path_to_bbox_validation = os.path.join(
                self.imagenet_dir_path, "ILSVRC2012_bbox_val_v3")
        elif self.series == "DET":
            path_to_img_validation = os.path.join(
                self.imagenet_dir_path, "ILSVRC2013_DET_val")
            path_to_bbox_validation = os.path.join(
                self.imagenet_dir_path, "ILSVRC2013_DET_bbox_val")
        else:
            self.error("Wrong series. Please choose DET or img")
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
                    os.mkdir(path_img_valid_dir, mode=0o775)
                except:
                    pass
                try:
                    os.mkdir(path_xml_valid_dir, mode=0o775)
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
            IMAGENET_BASE_PATH, "%s_%s_split_%s" % (self.year, self.series,
                                                    self.stage))
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

    def load_indices_hierarchy(self):
        hierarchy = {}
        with open(INDICES_HIERARCHY_FILE) as file:
            lines = file.read().splitlines()
            for line in lines:
                category_and_subcategories = line.split(" ")
                hierarchy[category_and_subcategories[0]] = \
                    category_and_subcategories[1:]

        return hierarchy

    def get_subcategories(self, hierarchy, index, recursive=False,
                          include_parent=False):
        subcategories = []
        if include_parent:
            subcategories += [index]
        if not recursive:
            subcategories += hierarchy[index]
            return subcategories
        else:
            stack = [index]
            while(len(stack) > 0):
                cur_category = stack.pop()
                new_subcategories = hierarchy[cur_category]
                stack += new_subcategories
                subcategories += new_subcategories

            return subcategories

    def parsing_and_split_DET_datset(self, path):
        self.split_dir = path
        bbox_path = os.path.join(self.split_dir, "ILSVRC2014_DET_bbox_train")
        img_path = os.path.join(self.split_dir, "ILSVRC2014_DET_train")
        self.year = "DET_dataset"
        self.series = "DET"
        set_type = "train"
        fnme = os.path.join(os.path.join(self.split_dir, IMAGES_JSON %
                            (self.year, self.series, set_type, self.stage)))
        try:
            with open(fnme, 'r') as fp:
                self.images_json[set_type] = json.load(fp)
        except:
            self.exception("Failed to load %s", fnme)
        self.init_files(self.split_dir)
        self.classes = []
        self.year = "2014"
        self.count_classes = 0
        self.imagenet_dir = os.path.join(IMAGENET_BASE_PATH, self.year)
        class_path = os.path.join(self.imagenet_dir, "ILSVRC2013_DET_val")
        for _root_path, sub_dirs, _files in os.walk(class_path,
                                                    followlinks=True):
            for sub_dir in sub_dirs:
                if sub_dir.find("none") == (-1):
                    self.classes.append(sub_dir)
                    self.count_classes += 1
        self.classes.sort()
        digits_word = []
        classes_word = []
        self.info("self.count_classes %s" % self.count_classes)
        categories_path = os.path.join(self.imagenet_dir,
                                       "indices_to_categories.txt")
        self.categories = open(categories_path, "r")
        for line in self.categories:
            digits_label = line[:line.find("\t")]
            word_label = line[line.find("\t") + 1:line.find("\n")]
            digits_word.append((digits_label, word_label))
        self.categories.close()
        digits_word.sort()
        ind = 0
        for label in self.classes:
            for (digits_label, word_label) in digits_word:
                if label == digits_label:
                    ind += 1
                    classes_word.append((ind, word_label))
        classes_word_path = os.path.join(
            self.imagenet_dir, "classes_200_%s_%s_%s_%s.json" %
            (self.year, self.series, set_type, self.stage))
        with open(classes_word_path, 'w') as fp:
            json.dump(classes_word, fp)
        if self.count_classes:
            for i in range(0, self.count_classes):
                try:
                    os.mkdir(os.path.join(bbox_path,
                                          "%s" % self.classes[i]), mode=0o775)
                except:
                    pass
                try:
                    os.mkdir(os.path.join(img_path,
                                          "%s" % self.classes[i]), mode=0o775)
                except:
                    pass
                try:
                    os.mkdir(os.path.join(img_path,
                                          "bbox_is_none"), mode=0o775)
                except:
                    pass
                try:
                    os.mkdir(os.path.join(bbox_path, "bbox_is_none"),
                             mode=0o775)
                except:
                    pass
            for image_name, _val in sorted(self.images_json[set_type].items()):
                image_path = self.images_json[set_type][image_name]["path"]
                bbox_name = image_name.replace(".JPEG", ".xml")
                bbox_name_dir_ind = image_path[:image_path.rfind("/")]
                bbox_name_dir = bbox_name_dir_ind[bbox_name_dir_ind.rfind("/")
                                                  + 1:]
                bbx_path = os.path.join(bbox_path, bbox_name_dir)
                bbx_path = os.path.join(bbx_path, bbox_name)
                ind = 0
                if self.images_json[set_type][image_name]["bbxs"] == []:
                    try:
                        shutil.copyfile(image_path, os.path.join(
                            img_path, "bbox_is_none/%s" % (image_name)))
                    except:
                        pass
                    try:
                        shutil.copyfile(
                            bbx_path, os.path.join(
                                bbox_path, "bbox_is_none/%s" % (bbox_name)))
                    except:
                        pass
                for bbx in self.images_json[set_type][image_name]["bbxs"]:
                    label = bbx["label"]
                    try:
                        shutil.copyfile(image_path, os.path.join(
                            img_path, "%s/%s" % (label, image_name)))
                    except:
                        pass
                    name_bbx = image_name.replace(".JPEG", ".xml")
                    try:
                        shutil.copyfile(
                            bbx_path, os.path.join(
                                bbox_path, "%s/%s" % (label, name_bbx)))
                    except:
                        pass
                    ind += 1
        hierarchy_det = {}
        hierarchy = self.load_indices_hierarchy()
        class_sub = 0
        for class_lbl in self.classes:
            if class_lbl != "negative_image":
                subcategories = self.get_subcategories(
                    hierarchy, class_lbl, recursive=True, include_parent=True)
                for _i in subcategories:
                    class_sub += 1
                hierarchy_det[class_lbl] = subcategories
        self.year = "2014"
        self.count_classes = 0
        self.imagenet_dir = os.path.join(IMAGENET_BASE_PATH, self.year)
        fnme = os.path.join(os.path.join(self.imagenet_dir, IMAGES_JSON %
                            (self.year, self.series, set_type, self.stage)))
        try:
            with open(fnme, 'r') as fp:
                self.images_json[set_type] = json.load(fp)
        except:
            self.exception("Failed to load %s", fnme)
        self.classes_sub = []
        sub_class_path = os.path.join(self.imagenet_dir,
                                      "ILSVRC2014_DET_train")
        for root_path, sub_dirs, files in os.walk(sub_class_path,
                                                  followlinks=True):
            for sub_dir in sub_dirs:
                if sub_dir.find("train") == (-1):
                    self.classes_sub.append(sub_dir)
        self.classes_sub.sort()
        hierarchy_rev = {}
        no_common_class = []
        for class_sub in self.classes_sub:
            self.no_common = True
            for class_com, value_list in sorted(hierarchy_det.items()):
                for value in value_list:
                    if class_sub == value:
                        self.no_common = False
                        hierarchy_rev[class_sub] = class_com
            if self.no_common:
                do_append = True
                for no_com in no_common_class:
                    if class_sub == no_com:
                        do_append = False
                if do_append:
                    no_common_class.append(class_sub)
        bbox_path = os.path.join(self.imagenet_dir,
                                 "ILSVRC2014_DET_bbox_train")
        for class_no in no_common_class:
            xml_class_path = os.path.join(bbox_path, class_no)
            for root_path, _tmp, files in os.walk(xml_class_path,
                                                  followlinks=True):
                for xml_class in files:
                    if xml_class.endswith(".xml"):
                        xml_path = os.path.join(root_path, xml_class)
                        with open(xml_path, "r") as fr:
                            tree = xmltodict.parse(fr.read())
                        if tree["annotation"].get("object") is not None:
                            temp_bbx = tree["annotation"]["object"]
                            if type(temp_bbx) is not list:
                                temp_bbx = [temp_bbx]
                            for bbx in temp_bbx:
                                class_common = bbx["name"]
                                hierarchy_rev[class_no] = class_common
        self.info("hierarchy_rev %s" % hierarchy_rev)
        self.info("len hierarchy_rev %s" % str(len(hierarchy_rev)))
        hierarchy_path = os.path.join(
            self.imagenet_dir, "hierarchy_%s_%s_%s_%s.json" %
            (self.year, self.series, set_type, self.stage))
        with open(hierarchy_path, 'w') as fp:
            json.dump(hierarchy_rev, fp)
        for class_sub, class_common in sorted(hierarchy_rev.items()):
            if class_common != "negative_image":
                if class_sub != class_common:
                    path_bbx_to = os.path.join(
                        self.split_dir, "ILSVRC2014_DET_bbox_train/%s/"
                        % class_common)
                    path_img_to = os.path.join(
                        self.split_dir, "ILSVRC2014_DET_train/%s/"
                        % class_common)
                    try:
                        os.mkdir(path_img_to, mode=0o775)
                    except:
                        pass
                    try:
                        os.mkdir(path_bbx_to, mode=0o775)
                    except:
                        pass
                else:
                    path_bbx_to = os.path.join(
                        self.split_dir, "ILSVRC2014_DET_bbox_train/")
                    path_img_to = os.path.join(
                        self.split_dir, "ILSVRC2014_DET_train/")
                path_bbx_from = os.path.join(
                    self.imagenet_dir,
                    "ILSVRC2014_DET_bbox_train/%s" % class_sub)
                path_img_from = os.path.join(
                    self.imagenet_dir, "ILSVRC2014_DET_train/%s" % class_sub)
                self.info("path_img_from %s" % path_img_from)
                self.info("path_img_to %s" % path_img_to)
                if os.system("rsync -aq '%s' '%s'" %
                             (path_img_from, path_img_to)):
                    raise RuntimeError("rsync failed")
                if os.system("rsync -aq '%s' '%s'" %
                             (path_bbx_from, path_bbx_to)):
                    raise RuntimeError("rsync failed")

    def min_max_shape(self, path):
        self.imagenet_dir_path = path
        max_shape = 0
        min_shape = 100000000
        for set_type in ("test", "validation", "train"):
            fnme = os.path.join(
                self.imagenet_dir_path, IMAGES_JSON %
                (self.year, self.series, set_type, self.stage))
            try:
                with open(fnme, 'r') as fp:
                    self.images_json[set_type] = json.load(fp)
            except:
                self.exception("Failed to load %s", fnme)
            for f, _val in sorted(self.images_json[set_type].items()):
                image_fnme = self.images_json[set_type][f]["path"]
                image = self.decode_image(image_fnme)
                _width = image.shape[1]
                _height = image.shape[0]
                for bbx in self.images_json[set_type][f]["bbxs"]:
                    bbx_height = bbx["height"]
                    bbx_width = bbx["width"]
                    shape = bbx_height * bbx_width
                    if shape < min_shape:
                        min_shape = shape
                        min_height = bbx_height
                        min_width = bbx_width
                    if shape > max_shape:
                        max_shape = shape
                        max_height = bbx_height
                        max_width = bbx_width
        self.info("min_shape w %s h %s" % (min_width, min_height))
        self.info("max_shape w %s h %s" % (max_width, max_height))

    def init_split_dataset(self, count_dirs):
        self.common_split_dir = os.path.join(
            IMAGENET_BASE_PATH, "%s_%s_split_%s" % (self.year, self.series,
                                                    self.stage))
        for i_patch in range(1, count_dirs + 1):
            self.year = str(i_patch)
            path_to_patch_folder = os.path.join(
                self.common_split_dir, "%s" % (i_patch))
            self.info("run init_files in %s" % path_to_patch_folder)
            self.init_files(path_to_patch_folder)

    def resize_split_dataset(self, count_dirs):
        self.common_split_dir = os.path.join(
            IMAGENET_BASE_PATH, "%s_%s_split_%s" % (self.year, self.series,
                                                    self.stage))
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

    def remove_background_split_dataset(self, count_dirs):
        self.common_split_dir = os.path.join(
            IMAGENET_BASE_PATH, "%s_split_%s" % (self.year, self.stage))
        for i_patch in range(1, count_dirs + 1):
            self.year = str(i_patch)
            path_to_patch_folder = os.path.join(
                self.common_split_dir, "%s" % (i_patch))
            self.info("run remove_background in %s"
                      % path_to_patch_folder)
            self.remove_background(path_to_patch_folder)

    def test_segmentation(self):
        src = "/home/lpodoynitsina/Desktop/image7.JPEG"
        """
        img = cv2.imread(src)
        do_gray = False
        if do_gray:
            img = cv2.imread(src, 0)
            img = cv2.medianBlur(img, 5)

            _ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
            th3 = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2)

            plt.subplot(2, 2, 1), plt.imshow(img, 'gray')
            plt.title('input image')
            plt.subplot(2, 2, 2), plt.imshow(th1, 'gray')
            plt.title('Global Thresholding')
            plt.subplot(2, 2, 3), plt.imshow(th2, 'gray')
            plt.title('Adaptive Mean Thresholding')
            plt.subplot(2, 2, 4), plt.imshow(th3, 'gray')
            plt.title('Adaptive Gaussian Thresholding')
        else:
            _ret, _thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            _ret, _thresh2 = cv2.threshold(img, 127, 255,
                                           cv2.THRESH_BINARY_INV)
            _ret, _thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
            _ret, _thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
            _ret, _thresh5 = cv2.threshold(img, 127, 255,
                                           cv2.THRESH_TOZERO_INV)

            thresh = ['img', '_thresh1', '_thresh2', '_thresh3', '_thresh4',
                      '_thresh5']

            for i in range(0, 6):
                plt.subplot(2, 3, i + 1), plt.imshow(eval(thresh[i]), 'gray')
                plt.title(thresh[i])
        """
        img = cv2.imread(src, 0)
        _ret1, _th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        _ret2, _th2 = cv2.threshold(img, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _ret3, _th3 = cv2.threshold(blur, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        titles = ['img', 'histogram1', '_th1',
                  'img', 'histogram2', '_th2',
                  'blur', 'histogram3', '_th3']
        blur_path = "/data/veles/tmp/bad_image.pickle"
        with open(blur_path, "wb") as fout:
            self.info("Saving labels of images to %s" %
                      blur_path)
            pickle.dump(blur.ravel(), fout)
        for i in range(0, 3):
            plt.subplot(3, 3, i * 3 + 1)
            plt.imshow(eval(titles[i * 3]), 'gray')
            plt.title(titles[i * 3])
            plt.subplot(3, 3, i * 3 + 2)
            plt.hist(eval(titles[i * 3]).ravel(), 256)
            plt.title(titles[i * 3 + 1])
            plt.subplot(3, 3, i * 3 + 3)
            plt.imshow(eval(titles[i * 3 + 2]), 'gray')
            plt.title(titles[i * 3 + 2])
        plt.show()

    def generate_negative_images(self, path):
        self.imagenet_dir_path = path
        min_size_max_side = 128
        max_count_neg_in_class = 5
        count_negative_in_class = 0
        class_is_new = False
        prev_label = ""
        rand = rnd.get()
        path_DET_train = os.path.join(self.imagenet_dir_path,
                                      "ILSVRC2014_DET_train")
        if self.series == "DET":
            path_to_save_train = os.path.join(
                self.imagenet_dir_path,
                "ILSVRC2014_DET_train/n00000000")
            path_to_save_valid = os.path.join(
                self.imagenet_dir_path,
                "ILSVRC2013_DET_val/n00000000")
            path_to_save_test = os.path.join(
                self.imagenet_dir_path,
                "ILSVRC2013_DET_test/n00000000")
            try:
                os.mkdir(path_to_save_train)
            except:
                pass
            try:
                os.mkdir(path_to_save_valid)
            except:
                pass
            try:
                os.mkdir(path_to_save_test)
            except:
                pass
            path_to_save_dict = {"train": path_to_save_train,
                                 "validation": path_to_save_valid,
                                 "test": path_to_save_test}
            file_to_open = os.path.join(
                IMAGENET_BASE_PATH,
                "2014/ILSVRC2014_devkit/data/det_lists/train_partall.txt")
            self.info("file_to_open %s" % file_to_open)
            part_images = []
            with open(file_to_open, "r") as file_part:
                for line in file_part:
                    path_for_part_img = os.path.join(
                        path_DET_train, line[:line.find("\n")]) + ".JPEG"
                    part_images.append(path_for_part_img)
        for set_type in ("test", "validation", "train"):
            if self.series == "DET":
                path_to_save = path_to_save_dict[set_type]
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
                do_negative = True
                if self.series == "DET":
                    for image_part in part_images:
                        if image_fnme == image_part:
                            self.info("image_fnme %s" % image_fnme)
                            self.info("image_part %s" % image_part)
                            do_negative = False
                if self.series == "img":
                    path_to_save = image_fnme[:image_fnme.rfind("/")]
                    path_to_save = path_to_save[:path_to_save.rfind("/")]
                    path_to_save = os.path.join(path_to_save, "n00000000")
                    try:
                        os.mkdir(path_to_save)
                    except:
                        pass
                if do_negative:
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
                            Matr = cv2.getRotationMatrix2D((x, y),
                                                           360 - ang, 1)
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
                        if count_negative_in_class < max_count_neg_in_class:
                            for _ in range(16):
                                stripe = rand.randint(4)
                                if stripe == 0:
                                    x_neg = x_min / 2
                                    w_neg = w_size
                                    h_neg = h_size
                                    # w_neg = x_min
                                    # h_neg = w_neg * h_size / w_size
                                    if (w_neg < min_size_min_side or
                                            h_neg < min_size_min_side or w_neg
                                            > x_min or
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
                                    scipy.misc.imsave(path_to_save_neg,
                                                      sample_neg)
                                    break
                                if stripe == 1:
                                    y_neg = y_min / 2
                                    w_neg = w_size
                                    h_neg = h_size
                                    # h_neg = y_min
                                    # w_neg = h_neg * w_size / h_size
                                    if (w_neg < min_size_min_side or
                                            h_neg < min_size_min_side or w_neg
                                            > image.shape[1] or
                                            h_neg > y_min):
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
                                    scipy.misc.imsave(path_to_save_neg,
                                                      sample_neg)
                                    break
                                if stripe == 2:
                                    x_neg = (image.shape[1]
                                             - x_max) / 2 + x_max
                                    w_neg = w_size
                                    h_neg = h_size
                                    # w_neg = image.shape[1] - x_max
                                    # h_neg = w_neg * h_size / w_size
                                    if (w_neg < min_size_min_side or
                                            h_neg < min_size_min_side or w_neg
                                            > image.shape[1] - x_max or
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
                                    scipy.misc.imsave(path_to_save_neg,
                                                      sample_neg)
                                    break
                                if stripe == 3:
                                    y_neg = (image.shape[0]
                                             - y_max) / 2 + y_max
                                    w_neg = w_size
                                    h_neg = h_size
                                    # h_neg = image.shape[0] - y_max
                                    # w_neg = h_neg * w_size / h_size
                                    if (w_neg < min_size_min_side or
                                            h_neg < min_size_min_side or w_neg
                                            > image.shape[1] or
                                            h_neg > image.shape[0] - y_max):
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
                                    scipy.misc.imsave(path_to_save_neg,
                                                      sample_neg)
                                    break

    def resize(self, img, x, y, h, w):
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x_min + w
        y_max = y_min + h
        sample_neg = img[y_min:y_max, x_min:x_max]
        return sample_neg

    def generate_negative_DET(self):
        self.year = "2014"
        rand = rnd.get()
        self.count_negative = 0
        self.imagenet_dir_path = os.path.join(IMAGENET_BASE_PATH, self.year)
        path_to_save = os.path.join(
            IMAGENET_BASE_PATH,
            "DET_dataset/ILSVRC2014_DET_train/n00000000")
        path_DET_train = os.path.join(self.imagenet_dir_path,
                                      "ILSVRC2014_DET_train")
        path_DET_train_bbox = os.path.join(self.imagenet_dir_path,
                                           "ILSVRC2014_DET_bbox_train")
        ind = 0
        for i in range(1, 201):
            file_to_open = os.path.join(
                IMAGENET_BASE_PATH,
                "2014/ILSVRC2014_devkit/data/det_lists/train_pos_%s.txt" % i)
            self.info("file_to_open %s" % file_to_open)
            with open(file_to_open, "r") as file_positive:
                for line in file_positive:
                    name_image = line[line.find("/") + 1: line.find("\n")]
                    path_for_positive_img = os.path.join(
                        path_DET_train, line[:line.find("\n")]) + ".JPEG"
                    image = self.decode_image(path_for_positive_img)
                    path_for_positive_bbox = os.path.join(
                        path_DET_train_bbox, line[:line.find("\n")]) + ".xml"
                    with open(path_for_positive_bbox, "r") as fr:
                        tree = xmltodict.parse(fr.read())
                    if tree["annotation"].get("object") is not None:
                        temp_bbx = tree["annotation"]["object"]
                        if type(temp_bbx) is not list:
                            temp_bbx = [temp_bbx]
                        bbx_xmax = []
                        bbx_xmin = []
                        bbx_ymax = []
                        bbx_ymin = []
                        for bbx in temp_bbx:
                            bbx_xmax.append(int(bbx["bndbox"]["xmax"]))
                            bbx_xmin.append(int(bbx["bndbox"]["xmin"]))
                            bbx_ymax.append(int(bbx["bndbox"]["ymax"]))
                            bbx_ymin.append(int(bbx["bndbox"]["ymin"]))
                            w_size = (int(bbx["bndbox"]["xmax"])
                                      - int(bbx["bndbox"]["xmin"]))
                            h_size = (int(bbx["bndbox"]["ymax"])
                                      - int(bbx["bndbox"]["ymin"]))
                        x_max = max(bbx_xmax)
                        x_min = min(bbx_xmin)
                        y_min = min(bbx_ymin)
                        y_max = max(bbx_xmax)
                        size_negative = max(
                            x_min, y_min, image.shape[0] - y_max,
                            image.shape[1] - x_max)
                        if size_negative > 64:
                            self.count_negative += 1
                            for _ in range(16):
                                if x_min == size_negative:
                                    x_neg = x_min / 2
                                    w_neg = w_size
                                    h_neg = h_size
                                    if (w_neg < size_negative or
                                            h_neg < size_negative or w_neg
                                            > x_min or
                                            h_neg > image.shape[0]):
                                        continue
                                    y_neg = h_neg / 2 + (
                                        image.shape[0] - h_neg) * rand.rand()
                                    sample_neg = self.resize(
                                        image, x_neg, y_neg, h_neg, w_neg)
                                    path_to_save_neg = os.path.join(
                                        path_to_save,
                                        "negative_image_%s_%s.JPEG" %
                                        (ind, name_image))
                                    ind += 1
                                    scipy.misc.imsave(path_to_save_neg,
                                                      sample_neg)
                                    break
                                if y_min == size_negative:
                                    y_neg = y_min / 2
                                    w_neg = w_size
                                    h_neg = h_size
                                    if (w_neg < size_negative or
                                            h_neg < size_negative or w_neg
                                            > image.shape[1] or
                                            h_neg > y_min):
                                        continue
                                    x_neg = w_neg / 2 + (
                                        image.shape[1] - w_neg) * rand.rand()
                                    sample_neg = self.resize(
                                        image, x_neg, y_neg, h_neg, w_neg)
                                    path_to_save_neg = os.path.join(
                                        path_to_save,
                                        "negative_image_%s_%s.JPEG" %
                                        (ind, name_image))
                                    ind += 1
                                    scipy.misc.imsave(path_to_save_neg,
                                                      sample_neg)
                                    break
                                if image.shape[1] - x_max == size_negative:
                                    x_neg = (image.shape[1]
                                             - x_max) / 2 + x_max
                                    w_neg = w_size
                                    h_neg = h_size
                                    if (w_neg < size_negative or
                                            h_neg < size_negative or w_neg
                                            > image.shape[1] - x_max or
                                            h_neg > image.shape[0]):
                                        continue
                                    y_neg = h_neg / 2 + (
                                        image.shape[0] - h_neg) * rand.rand()
                                    sample_neg = self.resize(
                                        image, x_neg, y_neg, h_neg, w_neg)
                                    path_to_save_neg = os.path.join(
                                        path_to_save,
                                        "negative_image_%s_%s.JPEG"
                                        % (ind, name_image))
                                    ind += 1
                                    scipy.misc.imsave(path_to_save_neg,
                                                      sample_neg)
                                    break
                                if image.shape[1] - x_max == size_negative:
                                    y_neg = (image.shape[0]
                                             - y_max) / 2 + y_max
                                    w_neg = w_size
                                    h_neg = h_size
                                    if (w_neg < size_negative or
                                            h_neg < size_negative or w_neg
                                            > image.shape[1] or
                                            h_neg > image.shape[0] - y_max):
                                        continue
                                    x_neg = w_neg / 2 + (
                                        image.shape[1] - w_neg) * rand.rand()
                                    sample_neg = self.resize(
                                        image, x_neg, y_neg, h_neg, w_neg)
                                    path_to_save_neg = os.path.join(
                                        path_to_save,
                                        "negative_image_%s_%s.JPEG"
                                        % (ind, name_image))
                                    ind += 1
                                    scipy.misc.imsave(path_to_save_neg,
                                                      sample_neg)
                                    break
                    else:
                        self.error("NOT POSITIVE IMAGE! %s"
                                   % path_for_positive_bbox)
            self.info("self.count_negative %s" % self.count_negative)
            file_positive.close()

    def remove_background(self, path):
        self.imagenet_dir_path = path
        paths_to_neg_dst = []
        if self.series == "img":
            for set_type in ("test", "validation", "train"):
                fnme = os.path.join(
                    self.imagenet_dir_path, IMAGES_JSON %
                    (self.year, self.series, set_type, self.stage))
                try:
                    with open(fnme, 'r') as fp:
                        self.images_json[set_type] = json.load(fp)
                except:
                    self.exception("Failed to load %s", fnme)
                for f, _val in sorted(self.images_json[set_type].items()):
                    image_fnme = self.images_json[set_type][f]["path"]
                    path_to_neg = image_fnme[:image_fnme.rfind("/")]
                    path = path_to_neg[:path_to_neg.rfind("/")]
                    do_append = True
                    for path_neg_in in paths_to_neg_dst:
                        if (os.path.join(path, "n00000000"),
                                os.path.join(path,
                                             "bad_negative")) == path_neg_in:
                            do_append = False
                    if do_append:
                        paths_to_neg_dst.append(
                            (os.path.join(path, "n00000000"),
                             os.path.join(path, "bad_negative")))
        if self.series == "DET":
            path_to_neg_train = os.path.join(self.imagenet_dir_path,
                                             "ILSVRC2014_DET_train/n00000000")
            dst_train = os.path.join(self.imagenet_dir_path,
                                     "ILSVRC2014_DET_train/bad_negative")
            path_to_neg_valid = os.path.join(self.imagenet_dir_path,
                                             "ILSVRC2013_DET_val/n00000000")
            dst_valid = os.path.join(self.imagenet_dir_path,
                                     "ILSVRC2013_DET_val/bad_negative")
            paths_to_neg_dst.append((path_to_neg_train, dst_train))
            paths_to_neg_dst.append((path_to_neg_valid, dst_valid))
        for (path_to_neg, dst) in paths_to_neg_dst:
            self.info("path_to_neg %s" % path_to_neg)
            self.info("dst %s" % dst)
            for root_path, _tmp, files in os.walk(path_to_neg,
                                                  followlinks=True):
                # self.info("dst %s" % dst)
                try:
                    os.mkdir(dst, mode=0o775)
                except:
                    pass
                for f in files:
                    if os.path.splitext(f)[1] == ".JPEG":
                        f_path = os.path.join(root_path, f)
                        good_backgr = back_det.is_background(f_path, 8.0)
                        if not good_backgr:
                            # self.info("%s is bad background" % f_path)
                            os.rename(f_path, os.path.join(dst, f))
                        # else:
                            # self.info("%s is good background" % f_path)
        # self.remove_dir(dst)

    def remove_dir(self, path):
        for rt, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(rt, f))
            for d in dirs:
                shutil.rmtree(os.path.join(rt, d))
            os.removedirs(path)
            logging.info("Remove directory %s" % path)

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
        categories_path = ("/data/veles/datasets/imagenet/2014/"
                           + "indices_to_categories.txt")
        self.categories = open(categories_path, "r")
        for line in self.categories:
            digits_label = line[:line.find("\t")]
            word_label = line[line.find("\t") + 1:line.find("\n")]
            digits_word.append((digits_label, word_label))
        self.categories.close()
        digits_word.sort()
        #colors = ("red", "green", "blue", "yellow", "pink", "black", "white",
        #          "orange", "brown", "cyan")
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
                self.info("*****draw bbx in image %s *****" %
                          self.images_json[set_type][f]["path"])
                self.info("self.images_json[set_type][f][bbxs] %s"
                          % self.images_json[set_type][f]["bbxs"])
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
                    for dig_word in digits_word:
                        if dig_word[0] == label:
                            label_txt = dig_word[1]
                    #color = colors[numpy.random.randint(len(colors))]
                    color = "red"
                    draw.text((x_min + 5, y_min), label_txt, fill=color,
                              font=font)
                    draw.line((x_min, y_min, x_min, y_max),
                              fill=color,
                              width=2)
                    draw.line((x_min, y_min, x_max, y_min),
                              fill=color,
                              width=2)
                    draw.line((x_min, y_max, x_max, y_max),
                              fill=color,
                              width=2)
                    draw.line((x_max, y_min, x_max, y_max),
                              fill=color,
                              width=2)
                path_to_image = self.images_json[set_type][f]["path"]
                ind_path = path_to_image.rfind("/")
                try:
                    os.mkdir(path_to_image[:ind_path])
                except OSError:
                    pass
                path_to_image = path_to_image.replace(
                    self.year, "images_with_bboxes_%s" % self.year)
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
        self.background = args.background
        if self.command_to_run == "init":
            self.init_files(os.path.join(IMAGENET_BASE_PATH, self.year))
        elif self.command_to_run == "draw_bbox":
            self.generate_images_with_bbx(os.path.join(IMAGENET_BASE_PATH,
                                                       self.year))
        elif self.command_to_run == "resize":
            self.generate_resized_dataset(os.path.join(IMAGENET_BASE_PATH,
                                                       self.year))
        elif self.command_to_run == "resize_validation":
            self.resize_validation(os.path.join(IMAGENET_BASE_PATH,
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
        elif self.command_to_run == "split_train":
            self.parsing_and_split_DET_datset(
                os.path.join(IMAGENET_BASE_PATH, "DET_dataset"))
        elif self.command_to_run == "test_segmentation":
            self.test_segmentation()
        elif self.command_to_run == "min_max_shape":
            self.min_max_shape(os.path.join(IMAGENET_BASE_PATH, self.year))
        elif self.command_to_run == "generate_negative_DET":
            self.generate_negative_DET()
        elif self.command_to_run == "test_load":
            self.test_load_data(os.path.join(IMAGENET_BASE_PATH, self.year))
        elif self.command_to_run == "remove_back_split_dataset":
            self.remove_background_split_dataset(self.count_dirs)
        elif self.command_to_run == "remove_back":
            self.remove_background(os.path.join(IMAGENET_BASE_PATH, self.year))
        elif self.command_to_run == "visualize":
            self.visualize_snapshot(args.snapshot, 900)
        elif self.command_to_run == "none_bboxes":
            self.calculate_bbox_is_none(os.path.join(IMAGENET_BASE_PATH,
                                                     self.year))
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
