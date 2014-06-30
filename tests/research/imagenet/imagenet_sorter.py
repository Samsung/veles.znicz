#!/usr/bin/python3
# encoding: utf-8
"""
Created on Jun 26, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
'''
This script to work with Imagenet dataset.

'''
try:
    import argcomplete
except:
    pass
import argparse
#import json
from PIL import Image, ImageDraw
import logging
import os
import sys
#import shutil
from veles.znicz.external import xmltodict

from veles.logger import Logger

IMAGENET_BASE_PATH = "/data/veles/datasets/imagenet"

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
        Logger.__init__(self, **kwargs)
        self.images_0 = {
            "train": {},  # {fname: (path, label, bboxes[] )}
            "validation": {},
            "test": {}
            }

        self.bboxes_0 = {
            "train": [],  # [(label, x, y, w, h, angle_radians)]
            "validation": [],
            "test": []
            }

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
            for root, _tmp, files in os.walk(path, followlinks=True):
                #print("ROOT=", root)
                for f in files:
                    if os.path.splitext(f)[1] == ".JPEG":
                        f_path = os.path.join(root, f)
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
                        temp_images[f] = (f_path, label, bbx)
                    else:
                        self.warning("Unexpected file in dir %s", f)
            if dir_bboxes != "":
                path = os.path.join(self.imagenet_dir_path, dir_bboxes)
                self.info("Scanning xml %s...", path)
                for root, _tmp, files in os.walk(path, followlinks=True):
                    for f in files:
                        if os.path.splitext(f)[1] == ".xml":
                            image_fname = os.path.splitext(f)[0] + ".JPEG"
                            xml_path = os.path.join(root, f)
                            with open(xml_path, "r") as fr:
                                tree = xmltodict.parse(fr.read())
                            #print("tree", tree)
                            bbx_labels = []
                            if type(tree["annotation"]["object"]) is list:
                                for i in range(0,
                                               len(tree["annotation"]["object"]
                                                   )):
                                    bbx_lbl = tree[
                                        "annotation"]["object"][i]["name"]
                                    bbx_xmax = int(
                                        tree["annotation"][
                                            "object"][i]["bndbox"]["xmax"])
                                    bbx_xmin = int(
                                        tree["annotation"][
                                            "object"][i]["bndbox"]["xmin"])
                                    bbx_ymax = int(
                                        tree["annotation"][
                                            "object"][i]["bndbox"]["ymax"])
                                    bbx_ymin = int(
                                        tree["annotation"][
                                            "object"][i]["bndbox"]["ymin"])
                                    bbx_ang = 0
                                    dict_bbx = {"label": bbx_lbl,
                                                "angle": bbx_ang,
                                                "xmin": bbx_xmin,
                                                "xmax": bbx_xmax,
                                                "ymin": bbx_ymin,
                                                "ymax": bbx_ymax}
                                    self.images_0[set_type][
                                        image_fname][2].append(dict_bbx)
                                    bbx_labels.append(bbx_lbl)
                            else:
                                bbx_lbl = tree["annotation"]["object"]["name"]
                                bbx_xmax = int(
                                    tree["annotation"][
                                        "object"]["bndbox"]["xmax"])
                                bbx_xmin = int(
                                    tree["annotation"][
                                        "object"]["bndbox"]["xmin"])
                                bbx_ymax = int(
                                    tree["annotation"][
                                        "object"]["bndbox"]["ymax"])
                                bbx_ymin = int(
                                    tree["annotation"][
                                        "object"]["bndbox"]["ymin"])
                                bbx_ang = 0
                                dict_bbx = {"label": bbx_lbl,
                                            "angle": bbx_ang,
                                            "xmin": bbx_xmin,
                                            "xmax": bbx_xmax,
                                            "ymin": bbx_ymin,
                                            "ymax": bbx_ymax}
                                self.images_0[
                                    set_type][image_fname][2].append(dict_bbx)
                                bbx_labels.append(bbx_lbl)
                            image_label = self.images_0[
                                set_type][image_fname][1]
                            for bbx_label in bbx_labels:
                                if bbx_label != image_label:
                                    label_bad = True
                                else:
                                    label_bad = False
                                    break
                            if label_bad is True:
                                self.info("label img %s"
                                          "not equal bbx_labels %s"
                                          % (image_label, bbx_labels))
        self.generate_images_with_bbx()
        return None

    def generate_images_with_bbx(self):
        self.imagenet_dir_path = "%s/%s" % (IMAGENET_BASE_PATH, self.year)
        """
        try:
            shutil.copytree(self.imagenet_dir_path,
                            os.path.join(IMAGENET_BASE_PATH,
                                         "images_with_bb %s" % self.year))
        except shutil.Error as e:
            self.info('Directory not copied. Error: %s' % e)
        """
        map_items = MAPPING[self.year][self.series].items()
        for set_type, (dir_images, _dir_bboxes) in map_items:
            path = os.path.join(self.imagenet_dir_path, dir_images)
            for _root, _tmp, files in os.walk(path, followlinks=True):
                for f in files:
                    image = Image.open(self.images_0[set_type][f][0])
                    draw = ImageDraw.Draw(image)
                    for i in range(0, len(self.images_0[set_type][f][2])):
                        x_min = self.images_0[set_type][f][2][i]["xmin"]
                        x_max = self.images_0[set_type][f][2][i]["xmax"]
                        y_min = self.images_0[set_type][f][2][i]["ymin"]
                        y_max = self.images_0[set_type][f][2][i]["ymax"]
                        self.info("*****draw*****")
                        draw.line((x_min, y_min, x_min, y_max),
                                  fill="green", width=3)
                        draw.line((x_min, y_min, x_max, y_min),
                                  fill="green", width=3)
                        draw.line((x_min, y_max, x_max, y_max),
                                  fill="green", width=3)
                        draw.line((x_max, y_min, x_max, y_max),
                                  fill="green", width=3)
                    path_to_image = self.images_0[set_type][f][0]
                    ind_path = path_to_image.rfind("/")
                    try:
                        os.mkdir(path_to_image[:ind_path])
                    except OSError:
                        pass
                    #file_name = path_to_image[ind_path + 1:]
                    path_to_image = path_to_image.replace("temp",
                                                          "images_with_bb")
                    image.save(path_to_image, "JPEG")

    def run(self):
        """Image net utility
        """
        parser = Main.init_parser()
        args = parser.parse_args()
        print(args)
        self.setup(level=Main.LOG_LEVEL_MAP[args.verbose])
        self.year = args.year
        self.series = args.series

        self.init_files()

        self.info("End of job")
        return Main.EXIT_SUCCESS

if __name__ == "__main__":
    sys.exit(Main().run())
