# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 29, 2014

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
        self.labels_mapping = {}
        self.ignore_negative = kwargs.get("ignore_negative", True)
        self.demand("winners", "mode")

    def initialize(self, **kwargs):
        self._results = {}
        with open(self.labels_txt, "r") as txt:
            values = txt.read().split()
            self.labels_mapping.update(dict(zip(map(int, values[::2]),
                                                values[1::2])))

    def run(self):
        """Winners must be of the format: {"path": ..., "bbxs": [...]}
        Each bbox is (confidence, label, (ymin, xmin, ymax, xmax)) if
        mode is "merge" else (confidence, label, {x, y, width, height)).
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
                if self.mode == "merge":
                    coords = bbox[2]
                    height, width = coords[2] - coords[0], \
                        coords[3] - coords[1]
                    assert width > 0
                    assert height > 0
                    y, x = (coords[2] + coords[0]) / 2, \
                        (coords[3] + coords[1]) / 2
                    bboxes.append({
                        "conf": float(bbox[1]),
                        "label": self.labels_mapping[
                            bbox[0] + (1 if self.ignore_negative else 0)],
                        "angle": "0", "x": int(numpy.round(x)),
                        "y": int(numpy.round(y)),
                        "width": int(numpy.round(width)),
                        "height": int(numpy.round(height))
                    })
                elif self.mode == "final":
                    bboxes.append({
                        "conf": float(bbox[1]),
                        "label": self.labels_mapping[
                            bbox[0] + (1 if self.ignore_negative else 0)],
                        "angle": "0", "x": int(numpy.round(bbox[2][0])),
                        "y": int(numpy.round(bbox[2][1])),
                        "width": int(numpy.round(bbox[2][2])),
                        "height": int(numpy.round(bbox[2][3]))
                    })
                else:
                    assert False
            self._results[os.path.basename(fn)] = {
                "path": fn,
                "label": "",
                "width": shape[0], "height": shape[1],
                "bbxs": bboxes
            }
        with open(self.result_path, "w") as fout:
            json.dump(self._results, fout, indent=4)
        self.info("Wrote %s", self.result_path)
