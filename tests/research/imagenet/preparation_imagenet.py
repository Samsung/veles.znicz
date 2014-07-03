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
import jpeg4py
import json
import logging
import numpy
import pickle
import os
from PIL import Image, ImageDraw
import scipy.misc
#import shutil
import sys

import veles.config as config
import veles.formats as formats
from veles.logger import Logger
from veles.znicz.external import xmltodict


IMAGENET_BASE_PATH = os.path.join(config.root.common.test_dataset_root,
                                  "imagenet")
IMAGES_JSON = "images_imagenet_%s_%s_%s_0.json"  # year, series, set_type

MAPPING = {
    "temp": {
        "img": {
            "train": ("ILSVRC2012_img_train", "ILSVRC2012_bbox_train_v2"),
            "validation": ("ILSVRC2012_img_val", "ILSVRC2012_bbox_val_v2"),
            "test": ("ILSVRC2012_img_test", ""),
        },
        "DET": {
            "train": ("ILSVRC2014_DET_train", "ILSVRC2014_DET_bbox_train"),
            "validation": ("ILSVRC2013_DET_val", "ILSVRC2013_DET_bbox_val"),
            "test": ("ILSVRC2013_DET_test", ""),
        },
    },
    "2014": {
        "img": {
            "train": ("ILSVRC2012_img_train", "ILSVRC2012_bbox_train_v2"),
            "validation": ("ILSVRC2012_img_val", "ILSVRC2012_bbox_val_v2"),
            "test": ("ILSVRC2012_img_test", ""),
        },
        "DET": {
            "train": ("ILSVRC2014_DET_train", "ILSVRC2014_DET_bbox_train"),
            "validation": ("ILSVRC2013_DET_val", "ILSVRC2013_DET_bbox_val"),
            "test": ("ILSVRC2013_DET_test", ""),
        },
    }
}


class Main(Logger):
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    LOG_LEVEL_MAP = {"debug": logging.DEBUG, "info": logging.INFO,
                     "warning": logging.WARNING, "error": logging.ERROR}

    def __init__(self, **kwargs):
        self.imagenet_dir_path = None
        self.year = None
        self.series = None
        self.fnme = None
        self.images_json = {
            "train": {},  # dict: {"path", "label", "bbx": [{bbx}, {bbx}, ...]}
            "validation": {},
            "test": {}
            }
        self.names_labels = {
            "train": [],
            "validation": [],
            "test": []
            }
        self.do_save_resized_images = kwargs.get("do_save_resized_images",
                                                 True)
        self.rect = kwargs.get("rect", (256, 256))
        self.minibatch_data = formats.Vector()
        self.minibatch_indices = formats.Vector()
        self.normalize = kwargs.get("normalize", True)
        self._max_minibatch_size = kwargs.get("minibatch_size", 100)
        self._sobel_kernel_size = kwargs.get(
            "sobel_kernel_size",
            config.get(config.root.imagenet.sobel_ksize) or 5)
        self._crop_color = kwargs.get(
            "crop_color",
            config.get(config.root.imagenet.crop_color) or (64, 64, 64))
        self._colorspace = kwargs.get(
            "colorspace", config.get(config.root.imagenet.colorspace) or "RGB")
        if self._colorspace == "GRAY":
            self._crop_color = self._crop_color[0]
        self._include_derivative = kwargs.get(
            "derivative", config.get(config.root.imagenet.derivative) or True)
        self._force_reinit = kwargs.get(
            "force_reinit",
            config.get(config.root.imagenet.force_reinit) or False)
        Logger.__init__(self, **kwargs)
        self.images_0 = {
            "train": {},  # dict: {"path", "label", "bbx": [{bbx}, {bbx}, ...]}
            "validation": {},
            "test": {}
            }

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
        parser.add_argument("command_to_run", type=str, default="",
                            choices=["all", "draw_bbox", "resize", "init"],
                            help="run functions: 'all' run all functions,"
                                 "'draw_bbox' run function which generate"
                                 "image with bboxes, 'resize' run function"
                                 "which resized images to bboxes, 'init' run"
                                 " function which generate json file")
        try:
            class NoEscapeCompleter(argcomplete.CompletionFinder):
                def quote_completions(self, completions, *args, **kwargs):
                    return completions
            NoEscapeCompleter()(parser)  # pylint: disable=E1102
        except:
            pass
        return parser

    def init_files(self):
        self.imagenet_dir_path = "%s/%s" % (IMAGENET_BASE_PATH, self.year)
        self.info("Looking for images in %s:", self.imagenet_dir_path)
        # finding dirs for images and bboxes
        map_items = MAPPING[self.year][self.series].items()
        for set_type, (dir_images, dir_bboxes) in map_items:
            print("------", set_type, dir_images, dir_bboxes)
            path = os.path.join(self.imagenet_dir_path, dir_images)
            self.info("Scanning JPG %s...", path)
            temp_images = self.images_0[set_type]
            for root_path, _tmp, files in os.walk(path, followlinks=True):
                #print("ROOT=", root)
                for f in files:
                    if os.path.splitext(f)[1] == ".JPEG":
                        f_path = os.path.join(root_path, f)
                        #--------------------------------------------
                        # KGG check if dirs have duplicates filenames
                        # KGG it was checked - no diplicates; code commented
                        #temp_image = temp_images.get(f)
                        #if temp_image != None:
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
                            #print("tree", tree)
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
                                image_lbl = self.images_0[
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
                                        self.images_0[
                                            set_type][image_fname]["path"])
                                dict_bbx = {"label": label,
                                            "angle": bbx_ang,
                                            "width": w,
                                            "height": h,
                                            "x": x,
                                            "y": y}
                                self.images_0[set_type][
                                    image_fname]["bbxs"].append(dict_bbx)

            cached_data_fnme = os.path.join(IMAGENET_BASE_PATH, self.year)
            try:
                os.mkdir(cached_data_fnme)
            except OSError:
                pass
            fnme = os.path.join(cached_data_fnme,
                                IMAGES_JSON %
                                (self.year, self.series, set_type))
            # image - dict: "path_to_img", "label", "bbx": [{bbx}, {bbx}, ...]
            with open(fnme, 'w') as fp:
                json.dump(self.images_0[set_type], fp)

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

    def imagenet_image_saver(self, data_names):
        for set_type in ("test", "validation", "train"):
            for (image, name) in data_names[set_type]:
                out_dirs = {"test":
                            os.path.join(config.root.common.cache_dir,
                                         "tmpimg/imagenet_image_saver"
                                         "/test/%s.jpg" % name),
                            "validation":
                            os.path.join(config.root.common.cache_dir,
                                         "tmpimg/imagenet_image_saver"
                                         "/validation/%s.jpg" % name),
                            "train":
                            os.path.join(config.root.common.cache_dir,
                                         "tmpimg/imagenet_image_saver"
                                         "/train/%s.jpg" % name)}
                self.info("Saving image %s" % out_dirs[set_type])
                try:
                    scipy.misc.imsave(out_dirs[set_type], image)
                except OSError:
                    self.error("Could not save image to %s"
                               % (out_dirs[set_type]))

    def generate_resized_dataset(self):
        self.info("Resized dataset")
        self.sobel = {"train": [], "test": [], "validation": []}
        data_names = {"train": [], "test": [], "validation": []}
        cached_data_fnme = os.path.join(IMAGENET_BASE_PATH, self.year)
        names_labels_dir = os.path.join(cached_data_fnme,
                                        "names_labels_%s_%s_0.pickle" %
                                        (self.year, self.series))
        for set_type in ("test", "validation", "train"):
            fnme = os.path.join(cached_data_fnme,
                                IMAGES_JSON %
                                (self.year, self.series, set_type))
            original_data_dir = os.path.join(
                cached_data_fnme,
                "original_data_%s_%s_%s_0.dat" %
                (self.year, self.series, set_type))
            self.f_samples = open(original_data_dir, "wb")
            try:
                with open(fnme, 'r') as fp:
                    self.images_json[set_type] = json.load(fp)
            except:
                self.exception("Failed to load %s", fnme)
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
                    sample = self.sample_rect(image, x, y, h_size, w_size)
                    sample_sobel = self.preprocess_sample(sample)
                    sobel = sample_sobel[1]
                    sample = sample_sobel[0]
                    sample.tofile(self.f_samples)
                    name_image = f[:f.rfind(".")]
                    name = name_image + ("_%s_bbx" % i)
                    self.names_labels[set_type].append((name, label))
                    data_names[set_type].append((sample, name))
                    self.sobel[set_type].append((sobel, name + "_sobel"))
                    i += 1
            self.info("Saving images to %s" % original_data_dir)
            self.f_samples.close()
        if self.do_save_resized_images:
            self.imagenet_image_saver(data_names)
            self.imagenet_image_saver(self.sobel)
        with open(names_labels_dir, "wb") as fout:
            self.info("Saving (name, label) of images to %s" %
                      names_labels_dir)
            pickle.dump(self.names_labels, fout)

    def decode_image(self, file_name):
        try:
            data = jpeg4py.JPEG(file_name).decode()
        except jpeg4py.JPEGRuntimeError as e:
            try:
                data = numpy.array(Image.open(file_name).convert("RGB"))
                self.warning("Falling back to PIL with file %s: %s",
                             file_name, repr(e))
            except:
                self.exception("Failed to decode %s", file_name)
                raise
        return data

    def sample_rect(self, img, x_c, y_c, h_size, w_size):
        x_min = x_c - w_size / 2
        y_min = y_c - h_size / 2
        x_max = x_min + w_size
        y_max = y_min + h_size
        bbox = [x_min, y_min, x_max, y_max]
        image_out = self.bbox_is_square(bbox, img)

        return image_out

    def bbox_is_square(self, bbox, img):
        width = img.shape[1]
        height = img.shape[0]
        offset = (bbox[2] - bbox[0] - (bbox[3] - bbox[1])) / 2
        if offset > 0:
            # Width is bigger than height
            bbox[1] -= int(numpy.floor(offset))
            bbox[3] += int(numpy.ceil(offset))
            bottom_height = -bbox[1]
            if bottom_height > 0:
                bbox[1] = 0
            else:
                bottom_height = 0
            top_height = bbox[3] - height
            if top_height > 0:
                bbox[3] = height
            else:
                top_height = 0
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            if bottom_height > 0:
                fixup = numpy.empty((bottom_height, bbox[2] - bbox[0], 3),
                                    dtype=img.dtype)
                fixup[:, :, :] = self._crop_color
                img = numpy.concatenate((fixup, img), axis=0)
            if top_height > 0:
                fixup = numpy.empty((top_height, bbox[2] - bbox[0], 3),
                                    dtype=img.dtype)
                fixup[:, :, :] = self._crop_color
                img = numpy.concatenate((img, fixup), axis=0)
        elif offset < 0:
            # Height is bigger than width
            bbox[0] += int(numpy.ceil(offset))
            bbox[2] -= int(numpy.floor(offset))
            left_width = -bbox[0]
            if left_width > 0:
                bbox[0] = 0
            else:
                left_width = 0
            right_width = bbox[2] - width
            if right_width > 0:
                bbox[2] = width
            else:
                right_width = 0
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            if left_width > 0:
                fixup = numpy.empty((bbox[3] - bbox[1], left_width, 3),
                                    dtype=img.dtype)
                fixup[:, :, :] = self._crop_color
                img = numpy.concatenate((fixup, img), axis=1)
            if right_width > 0:
                fixup = numpy.empty((bbox[3] - bbox[1], right_width, 3),
                                    dtype=img.dtype)
                fixup[:, :, :] = self._crop_color
                img = numpy.concatenate((img, fixup), axis=1)
        else:
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        assert img.shape[0] == img.shape[1]
        if img.shape[0] != self.rect[0]:
            img = cv2.resize(img, self.rect,
                             interpolation=cv2.INTER_AREA)
        return img

    def generate_images_with_bbx(self):
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
        cached_data_fnme = os.path.join(IMAGENET_BASE_PATH, self.year)
        for set_type in ("test", "validation", "train"):
            fnme = os.path.join(cached_data_fnme,
                                IMAGES_JSON %
                                (self.year, self.series, set_type))
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
                    x_min = x - w / 2
                    y_min = y - h / 2
                    x_max = x_min + w
                    y_max = y_min + h
                    self.info("*****draw bbx in image %s *****" %
                              self.images_json[set_type][f]["path"])
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
        if self.command_to_run == "all":
            self.init_files()
            self.generate_images_with_bbx()
            self.generate_resized_dataset()
        elif self.command_to_run == "init":
            self.init_files()
        elif self.command_to_run == "draw_bbox":
            self.generate_images_with_bbx()
        elif self.command_to_run == "resize":
            self.generate_resized_dataset()
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
