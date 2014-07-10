"""
Created on Jul 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import json
import os

from veles.config import root
from veles.znicz.loader import Loader

IMAGES_JSON = "images_imagenet_%s_%s_%s_%s.json"
# year, series, set_type, iteration
IMAGENET_BASE_PATH = os.path.join(root.common.test_dataset_root,
                                  "imagenet")

root.defaults = {"forward": {"year": "temp",
                             "iteration": 0,
                             "series": "img"}}
root.forward.imagenet_dir_path = os.path.join(IMAGENET_BASE_PATH,
                                              root.forward.year)


class ForwardStage1(Loader):
    """
    Imagenet loader for first processing stage.
    """

    def __init__(self, workflow, **kwargs):
        super(ForwardStage1, self).__init__(workflow, **kwargs)
        self.year = root.forward.year
        self.iteration = root.forward.iteration
        self.series = root.forward.series
        self.images_json = {
            "test": {},  # dict: {"path", "label", "bbx": [{bbx}, {bbx}, ...]}
            "validation": {},
            "train": {}
            }

    def calculate_threshold(self):
        self.imagenet_dir_path = root.forward.imagenet_dir_path
        for set_type in ("test", "validation", "train"):
            fnme = os.path.join(
                self.imagenet_dir_path, IMAGES_JSON %
                (self.year, self.series, set_type, self.iteration))
            try:
                self.info("Loading images info from %s to calculate threshold"
                          % fnme)
                with open(fnme, 'r') as fp:
                    self.images_json[set_type] = json.load(fp)
            except:
                self.exception("Failed to load %s", fnme)
            for image_name, _val in sorted(self.images_json[set_type].items()):
                image_fnme = self.images_json[set_type][image_name]["path"]
                _image = self.decode_image(image_fnme)
                i = 0
                for bbx in self.images_json[set_type][image_name]["bbxs"]:
                    _label = bbx["label"]
                    _x = bbx["x"]
                    _y = bbx["y"]
                    _h_size = bbx["height"]
                    _w_size = bbx["width"]
                    _ang = bbx["angle"]
                    _name_sample = (image_name[:image_name.rfind(".")] +
                                    ("_%s_bbx" % i))
