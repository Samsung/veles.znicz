#!/usr/bin/python3
# encoding: utf-8
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on July 4, 2014

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
import numpy
import sys
import os
from scipy.io import loadmat

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from veles.znicz.tests.research.imagenet.forward_bbox import BBox


class InvalidBBox(Exception):
    pass


def get_bbox_min_max(bbox, iwh):
    # our angle is always zero
    """
    angle = float(bbox["angle"])
    matrix = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                          [numpy.sin(angle), numpy.cos(angle)]])
    """
    w, h = bbox["width"], bbox["height"]
    x, y = bbox["x"], bbox["y"]
    if w <= 0 or h <= 0:
        raise InvalidBBox()
    bb = numpy.array([[x - w // 2, y - h // 2], [x - w // 2 + w, y - h // 2],
                      [x - w // 2 + w, y - h // 2 + h],
                      [x - w // 2, y - h // 2 + h]])
    """
    bb = bb.dot(matrix)
    """
    xmin, ymin = [max(numpy.min(bb[:, i]), 0) for i in (0, 1)]
    xmax, ymax = [min(numpy.max(bb[:, i]), iwh[i]) for i in (0, 1)]
    return xmin, ymin, xmax, ymax


def get_image_dims(val):
    res = [val[dim] for dim in ("width", "height")]
    if res[0] == res[1] == -1:
        return (100000, 100000)
    return res


def convert_DET(idk, dset, ijson, otxt):
    """
    *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    2.4 DET submission format
    *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Submission of the results will consist of a text file
    with one line per predicted object. It looks as follows:

    <image_index> <ILSVRC2014_DET_ID> <confidence> <xmin> <ymin> <xmax> <ymax>
    """
    img_file = os.path.join(idk, "data/det_lists/%s.txt" % ("val"
                            if dset == "validation" else "test"))
    with open(img_file, "r") as txt:
        values = txt.read().split()
        img_mapping = dict(zip(values[::2], map(int, values[1::2])))
    print("Read %d image indices" % len(img_mapping))
    labels = loadmat(os.path.join(idk, "data/meta_det.mat"))
    labels_mapping = {str(s[1][0]): int(s[0][0][0])
                      for s in labels['synsets'][0]}
    print("Read %d labels" % len(labels_mapping))
    for key, val in sorted(ijson.items()):
        if len(val["bbxs"]) == 0:
            print("Warning: %s has no bboxes" % key)
            continue
        iwh = get_image_dims(val)
        for bbox in val["bbxs"]:
            try:
                minmaxs = get_bbox_min_max(bbox, iwh)
            except InvalidBBox:
                print("Warning: %s has a bbox with an invalid width or "
                      "height: %s" % (key, bbox))
                continue
            try:
                otxt.write(("%d %d %.3f " + "%d " * 4 + "\n") % ((
                           img_mapping[os.path.splitext(key)[0]],
                           labels_mapping[bbox["label"]],
                           bbox["conf"]) + minmaxs))
            except KeyError:
                pass


def convert_CLS_LOC(idk, dset, ijson, otxt):
    """
    *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    3.3 CLS-LOC submission format
    *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    The submission of results on test data will consist of a text file
    with one line per image, in the alphabetical order of the image file
    names, i.e. from ILSVRC2012_test_00000001.JPEG to
    ILSVRC2012_test_0100000.JPEG. Each line contains up to 5 detected
    objects, sorted by confidence in descending order. The format is as
    follows:

        <label(1)> <xmin(1)> <ymin(1)> <xmax(1)> <ymax(1)> <label(2)> <xmin(2)>
            <ymin(2)> <xmax(2)> <ymax(2)> ....

    The predicted labels are the ILSVRC2014_IDs ( integers between 1 and
    1000 ).  The number of labels per line can vary, but not more than 5
    (extra labels are ignored).
    """

    val_size = 50000

    labels = loadmat(os.path.join(idk, "data/meta_clsloc.mat"))
    synset_indices = {}
    for s in labels["synsets"][0]:
        index, synset, name = s[0][0][0], s[1][0], s[2][0]
        synset_indices[synset] = index

    for i in range(val_size):
        pic_name = "ILSVRC2012_val_%.8d.JPEG" % (i + 1)
        line_to_write = ""
        if pic_name in ijson:
            bboxes = [x for x in ijson[pic_name]["bbxs"]
                      if x["label"] in synset_indices]
            for bbox in list(sorted(bboxes, key=lambda box: box["conf"],
                                    reverse=True))[:5]:
                line_to_write += str(synset_indices[bbox["label"]])
                box_obj = BBox.from_json_dict(bbox)
                line_to_write += (" %.0f %.0f %.0f %.0f " %
                                  (box_obj.xmin, box_obj.ymin,
                                   box_obj.xmax, box_obj.ymax))
        if line_to_write == "":
            line_to_write += "0 0 1 0 1"  # class None
        otxt.write(line_to_write)
        otxt.write("\n")


def main():
    parser = ArgumentParser(
        description="Convert Veles Imagenet JSON to txt submission format",
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--idk", help="Path to the Imagenet Development Kit.")
    parser.add_argument('input.json',
                        help='Path to the JSON file to convert.')
    args = parser.parse_args()
    idk = args.idk
    ifile = getattr(args, "input.json")
    base_file = os.path.basename(ifile)
    parsed = os.path.splitext(base_file)[0].split('_')
    modes = {"det": "DET", "img": "CLS-LOC"}
    mode = modes[parsed[-3]]
    dsets = {"validation": "validation", "test": "test"}
    dset = dsets[parsed[-2]]
    print("Detected challenge: %s (%s)" % (mode, dset))
    with open(ifile, "r") as json_file:
        ijson = json.load(json_file)
    print("Read %d files from %s" % (len(ijson), base_file))
    result_path = "%s.txt" % os.path.splitext(base_file)[0]
    with open(result_path, "w") as otxt:
        if mode == "DET":
            convert_DET(idk, dset, ijson, otxt)
        else:
            convert_CLS_LOC(idk, dset, ijson, otxt)
    print("Wrote %s" % result_path)

if __name__ == "__main__":
    sys.exit(main())
