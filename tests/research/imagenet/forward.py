"""
Created on Jul 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import json
import numpy

from veles import OpenCLUnit
import veles.formats as formats


class ForwardStage1Loader(OpenCLUnit):
    """
    Imagenet loader for first processing stage.
    """

    def __init__(self, workflow, images_json, imagenet_path, **kwargs):
        super(ForwardStage1Loader, self).__init__(workflow, **kwargs)
        self.images_json = images_json
        self.imagenet_path = imagenet_path
        self.images = {}
        self.mapping = {}
        self.angle_step = kwargs.get('angle_step', 0.087266)  # 2 * PI / 72
        self.scale_step = kwargs.get('scale_step', 0.1)
        self.max_scale_steps = kwargs.get('max_scale_steps', 100)
        self.stage = 1  # 1 or 2
        self.current_image = ""
        self.batch_data = formats.Vector()
        self.batch_size = 0
        self.batch_bboxes = formats.Vector()
        self.demand("entry")  # first forward unit

    def initialize(self, device, **kwargs):
        super(ForwardStage1Loader, self).initialize(**kwargs)
        for set_type in ("test", "validation", "train"):
            images_json = self.images_json % set_type
            try:
                self.info("Loading images JSON from %s" % images_json)
                with open(images_json, 'r') as fp:
                    self.images[set_type] = json.load(fp)
            except:
                self.exception("Failed to load %s", images_json)
            self.mapping[set_type] = {}
            for image_name, meta in sorted(self.images[set_type].items()):
                for bbx in meta["bbxs"]:
                    label = bbx["label"]
                    self.mapping[set_type][label].append(image_name)

        aperture_size = 256 * 256 * 4  # FIXME: get it from self.entry
        max_batch_size = int(2 * numpy.pi / self.angle_step) * \
            self.max_scale_steps
        self.batch_data.mem = numpy.zeros((max_batch_size, aperture_size),
                                          dtype=self.entry.weights.mem.dtype)
        self.batch_bboxes.mem = numpy.zeros(4 * max_batch_size,
                                            dtype=numpy.int32)

        if device is None:
            return

        self.batch_data.initialize(device)
        self.batch_bboxes.initialize(device)

    def calculate_scale_steps(self):
        pass

    def get_image_data(self, image_name):
        pass
