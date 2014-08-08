"""
Created on Jul 29, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import json
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
        """Winners must be of the format: {"path": ..., "bbxs": [...]}
        Each bbox is {"conf": %f, "label": %d, "angle": %f,
                      "bbox": (label_index, confidence,
                               (xmin, ymin, xmax, ymax))}.
        """
        if self.winners is None:
            return
        self.info("Writing the results for %d images...", len(self.winners))
        for win in self.winners:
            fn = win["path"]
            try:
                shape = Image.open(fn).size
            except:
                shape = (-1, -1)
                self.warning("Failed to determine the size of %s", fn)
            bboxes = []
            for bbox in win["bbxs"]:
                coords = bbox[2]
                width, height = coords[2] - coords[0], coords[3] - coords[1]
                x, y = (coords[2] + coords[0]) / 2, (coords[3] - coords[1]) / 2
                bboxes.append({
                    "conf": bbox[1],
                    "label": self._labels_mapping[bbox[0]],
                    "angle": "0", "x": x, "y": y,
                    "width": width, "height": height
                })
            self._results[os.path.basename(fn)] = {
                "path": fn,
                "label": "",
                "width": shape[0], "height": shape[1],
                "bbxs": bboxes
            }
        with open(self.result_path, "w") as fout:
            json.dump(self._results, fout, indent=4)
        self.info("Wrote %s", self.result_path)
