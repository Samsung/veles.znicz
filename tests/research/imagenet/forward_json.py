"""
Created on Jul 29, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import json
import numpy
import os
from PIL import Image
from zope.interface import implementer

from veles.units import Unit, IUnit


@implementer(IUnit)
class ImagenetResultWriter(Unit):
    """Writes JSON with resulting bboxes.
    """

    def __init__(self, workflow, labels_txt, result_path, **kwargs):
        super(ImagenetResultWriter, self).__init__(workflow, **kwargs)
        self.labels_txt = labels_txt
        self.result_path = result_path
        self.demand("winners")

    def initialize(self, **kwargs):
        self._results = {}
        with open(self.labels_txt, "r") as txt:
            values = txt.read().split()
            self._labels_mapping = dict(zip(map(int, values[::2]),
                                            values[1::2]))

    def run(self):
        """Winners must be of the format: {"path": ..., "bboxes": [...]}
        Each bbox is {"conf": %f, "label": %d, "angle": %f,
                      "bbox": numpy array of shape (4, 2)}.
        """
        if self.winners is None:
            return
        for win in self.winners:
            fn = win["path"]
            try:
                shape = Image.open(fn).size
            except:
                shape = (-1, -1)
                self.warning("Failed to determine the size of %s", fn)
            bboxes = []
            for bbox in win["bboxes"]:
                bb_coords = bbox["bbox"]
                angle = bbox["angle"]
                matrix = numpy.array([[numpy.cos(angle), numpy.sin(angle)],
                                      [-numpy.sin(angle), numpy.cos(angle)]])
                bb_rot = bb_coords.dot(matrix)
                width, height = (numpy.max(bb_rot[:, i]) -
                                 numpy.min(bb_rot[:, i]) for i in (0, 1))
                x, y = (numpy.mean(bb_coords[:, i]) for i in (0, 1))
                bboxes.append({
                    "conf": bbox["conf"],
                    "label": self._labels_mapping[bbox["label"]],
                    "angle": angle, "x": x, "y": y,
                    "width": width, "height": height
                })
            self._results[os.path.basename(fn)] = {
                "path": fn,
                "label": "",
                "width": shape[0], "height": shape[1],
                "bbxs": bboxes
            }

    def write(self):
        with open(self.result_path, "w") as fout:
            json.dump(self._results, fout, indent=4)
