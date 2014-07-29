#!/usr/bin/python3
# encoding: utf-8

import json
import numpy
import sys
import os
from scipy.io import loadmat

from argparse import ArgumentParser, RawDescriptionHelpFormatter


def get_bbox_min_max(bbox, iwh):
    angle = bbox["angle"]
    matrix = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                          [numpy.sin(angle), numpy.cos(angle)]])
    w, h = bbox["width"], bbox["height"]
    bb = numpy.array([[0, 0], [w, 0], [w, h], [0, h]])
    bb = bb.dot(matrix)
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
    img_file = os.path.join(idk, "data/det_lists/%s.txt" % "val"
                            if dset == "validation" else "test")
    with open(img_file, "r") as txt:
        values = txt.read().split()
        img_mapping = dict(zip(values[::2], map(int, values[1::2])))
    print("Read %d image indices" % len(img_mapping))
    labels = loadmat(os.path.join(idk, "data/meta_det.mat"))
    labels_mapping = {s[1]: s[0] for s in labels['synsets']}
    print("Read %d labels" % len(labels_mapping))
    for key, val in sorted(ijson.items()):
        bbox = max(val["bbxs"], key=lambda bbox: bbox["conf"])
        iwh = get_image_dims(val)
        minmaxs = get_bbox_min_max(bbox, iwh)
        otxt.write(("%d %d %.3f " + "%d " * 4 + "\n") % ((
                   img_mapping[os.path.splitext(key)[0]],
                   labels_mapping[bbox["label"]],
                   bbox["conf"]) + minmaxs))


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
    labels = loadmat(os.path.join(idk, "data/meta_clsloc.mat"))
    labels_mapping = {s[1]: s[0] for s in labels['synsets']}
    print("Read %d labels" % len(labels_mapping))
    for _, val in sorted(ijson.items()):
        bboxes = val["bbxs"]
        iwh = get_image_dims(val)
        for bbox in list(sorted(bboxes, key=lambda bbox: bbox["conf"],
                                reverse=True))[:5]:
            minmaxs = get_bbox_min_max(bbox, iwh)
            otxt.write(("%d " * 5 + "\n") % (
                (labels_mapping[bbox["label"]],) + minmaxs))


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
    with open("%s.txt" % os.path.splitext(base_file)[0], "w") as otxt:
        if mode == "DET":
            convert_DET(idk, dset, ijson, otxt)
        else:
            convert_CLS_LOC(idk, dset, ijson, otxt)

if __name__ == "__main__":
    sys.exit(main())
