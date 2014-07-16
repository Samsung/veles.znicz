"""
Created on Jul 16, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import numpy
import os
from PIL import Image
from zope.interface import implementer

from veles.units import Unit, IUnit
from .forward import ImagenetForwardLoader


@implementer(IUnit)
class ImagenetBboxMapper(Unit):
    """
    Draws maps of object "hotness".
    """

    def __init__(self, workflow, result_path, **kwargs):
        super(ImagenetBboxMapper, self).__init__(workflow, **kwargs)
        self.result_path = result_path
        self.current_map = None
        self.current_image_name = ""
        self.demand("classified", "minibatch_bboxes", "minibatch_image_names",
                    "minibatch_image_shapes", "labels_number")

    def initialize(self, **kwargs):
        pass

    def run(self):
        self.classified.map_read()

        for index in range(len(self.classified.mem.shape[0])):
            image_name = self.minibatch_image_names[index]
            if self.current_image_name != image_name:
                if not self.current_map is None:
                    self._save_current_map()
                self.current_map = numpy.zeros(
                    tuple(self.minibatch_image_shapes[index]) +
                    (self.labels_number,))
                self.current_image_name = image_name
            bbox = self.minibatch_bboxes[index]
            mins = [int(numpy.min(bbox[:, i])) for i in (0, 1)]
            maxs = [int(numpy.max(bbox[:, i])) for i in (0, 1)]
            for x in range(mins[0], maxs[0]):
                for y in range(mins[1], maxs[1]):
                    if ImagenetForwardLoader.inside((x, y), bbox):
                        for label in range(self.labels_number):
                            self.current_map[y, x, label] += \
                                self.classified.mem[index, label]

    def _save_current_map(self):
        base = os.path.join(self.result_path,
                            self.current_image_name + "%d.png")
        for label in range(self.labels_number):
            fn = base % label
            img = Image.fromarray(self.current_map[:, :, label], 'L')
            img.save(fn)
