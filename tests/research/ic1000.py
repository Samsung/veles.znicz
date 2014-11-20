#!/usr/bin/python3 -O
"""
Created on Nov 14, 2014

File for CaffeNet workflow (convolutional group=1 only).

Copyright (c) 2014 Samsung R&D Institute Russia
"""


import os

from veles.config import root
import veles.znicz.loader as loader

from veles.znicz.standard_workflow import StandardWorkflow

data_dir = "/data/veles/datasets/FakeImagenet"


class ImagenetLoader(loader.ImageLoader):
    def load_data(self):
        super(ImagenetLoader, self).load_data()
        self.extract_validation_from_train(
            ratio=root.imagenet.loader.validation_ratio)

    def get_label_from_filename(self, filename):
        dn = os.path.dirname(filename)
        return int(dn[dn.rfind("/") + 1:])

    def is_valid_filename(self, filename):
        return (os.path.splitext(filename)[1] == ".png")


class ImagenetWorkflow(StandardWorkflow):
    def __init__(self, workflow, **kwargs):
        super(ImagenetWorkflow, self).__init__(
            workflow,
            fail_iterations=root.imagenet.decision.fail_iterations,
            max_epochs=root.imagenet.decision.max_epochs,
            prefix=root.imagenet.snapshotter.prefix,
            interval=root.imagenet.snapshotter.interval,
            snapshot_dir=root.common.snapshot_dir,
            layers=root.imagenet.layers,
            loss_function=root.imagenet.loss_function, **kwargs)

    def link_loader(self):
        super(ImagenetWorkflow, self).link_loader()
        self.loader = ImagenetLoader(
            self, on_device=root.imagenet.loader.on_device,
            train_paths=[data_dir],
            minibatch_size=root.imagenet.loader.minibatch_size)
        self.loader.link_from(self.repeater)


def run(load, main):
    load(ImagenetWorkflow)
    main()
