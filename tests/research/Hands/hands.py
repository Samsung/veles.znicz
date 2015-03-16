#!/usr/bin/python3 -O
"""
Created on Jun 14, 2013

Model created for human hands recognition. Dataset - Samsung Database with
images of human hands.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
from zope.interface import implementer

from veles.config import root
from veles.loader import IFileImageLoader
from veles.loader.fullbatch_image import FullBatchAutoLabelFileImageLoader
from veles.znicz.standard_workflow import StandardWorkflow


@implementer(IFileImageLoader)
class HandsLoader(FullBatchAutoLabelFileImageLoader):
    """Loads Hands dataset.
    """
    MAPPING = "hands_loader"

    def get_image_data(self, key):
        a = numpy.fromfile(key, dtype=numpy.uint8).astype(numpy.float32)
        sx = int(numpy.sqrt(a.size))
        a = a.reshape(sx, sx).astype(numpy.float32)
        return a

    def get_image_info(self, key):
        img = numpy.fromfile(key, dtype=numpy.uint8).astype(numpy.float32)
        sx = int(numpy.sqrt(img.size))
        return (sx, sx), "GRAY"

    def is_valid_filename(self, filename):
        return filename[-4:] == ".raw"


class HandsWorkflow(StandardWorkflow):
    """Sample workflow for Hands dataset.
    """
    def create_workflow(self):
        self.link_repeater(self.start_point)

        self.link_loader(self.repeater)

        self.link_forwards(("input", "minibatch_data"), self.loader)

        self.link_evaluator(self.forwards[-1])

        self.link_decision(self.evaluator)

        end_units = tuple(link(self.decision) for link in (
            self.link_snapshotter, self.link_error_plotter,
            self.link_conf_matrix_plotter))

        self.link_gds(self.repeater, *end_units)

        self.link_end_point(*end_units)


def run(load, main):
    load(HandsWorkflow,
         decision_config=root.hands.decision,
         snapshotter_config=root.hands.snapshotter,
         loader_config=root.hands.loader,
         layers=root.hands.layers,
         loss_function=root.hands.loss_function,
         loader_name=root.hands.loader_name)
    main()
