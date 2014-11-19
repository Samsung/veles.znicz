#!/usr/bin/python3 -O
'''
Created on Nov 13, 2014

Model was created for face recognition. Database - Yale Faces.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
'''

import os
import wget
import zipfile

from veles.config import root
from veles.znicz.loader import ImageLoader
from veles.znicz.standard_workflow import StandardWorkflow

common_dir = root.common.test_dataset_root
data_dir = os.path.join(common_dir, "CroppedYale")

root.yalefaces.update({
    "decision": {"fail_iterations": 50, "max_epochs": 1000},
    "loss_function": "softmax",
    "snapshotter": {"prefix": "yalefaces"},
    "loader": {"minibatch_size": 40, "on_device": True,
               "validation_ratio": 0.15,
               "url":
               "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/"
               "CroppedYale.zip"},
    "layers": [{"type": "all2all_tanh", "learning_rate": 0.01,
                "weights_decay": 0.00005, "output_shape": 100},
               {"type": "softmax", "output_shape": 39, "learning_rate": 0.01,
                "weights_decay": 0.00005}]})


class YaleFacesLoader(ImageLoader):
    def load_data(self):
        if not os.path.exists(data_dir):
            url = root.yalefaces.loader.url
            self.warning("%s does not exist, downloading from %s...",
                         data_dir, url)
            dir_zip = wget.download("%s" % url, common_dir)
            try:
                with zipfile.ZipFile(dir_zip) as zip:
                    zip.extractall(common_dir)
            finally:
                os.remove(dir_zip)
        self.info("Loading from original Yale Faces files...")
        super(YaleFacesLoader, self).load_data()
        self.extract_validation_from_train(
            ratio=root.yalefaces.loader.validation_ratio)

    def get_label_from_filename(self, filename):
        dn = os.path.dirname(filename)
        return int(dn[-2:]) - 1

    def is_valid_filename(self, filename):
        return (os.path.splitext(filename)[1] == ".pgm"
                and filename.find("Ambient") < 0)


class YaleFacesWorkflow(StandardWorkflow):
    def __init__(self, workflow, **kwargs):
        super(YaleFacesWorkflow, self).__init__(
            workflow,
            fail_iterations=root.yalefaces.decision.fail_iterations,
            max_epochs=root.yalefaces.decision.max_epochs,
            prefix=root.yalefaces.snapshotter.prefix,
            snapshot_dir=root.common.snapshot_dir,
            layers=root.yalefaces.layers,
            loss_function=root.yalefaces.loss_function, **kwargs)

    def link_loader(self):
        super(YaleFacesWorkflow, self).link_loader()
        self.loader = YaleFacesLoader(
            self, on_device=root.yalefaces.loader.on_device,
            train_paths=[data_dir],
            minibatch_size=root.yalefaces.loader.minibatch_size)
        self.loader.link_from(self.repeater)


def run(load, main):
    load(YaleFacesWorkflow)
    main()
