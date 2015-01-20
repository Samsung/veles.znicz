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
from veles.znicz.loader import FullBatchImageLoader
from veles.znicz.standard_workflow import StandardWorkflow


class YaleFacesLoader(FullBatchImageLoader):
    def __init__(self, workflow, **kwargs):
        super(YaleFacesLoader, self).__init__(workflow, **kwargs)
        self.data_dir = kwargs.get("data_dir", "")
        self.common_dir = kwargs.get("common_dir", "")

    def load_data(self):
        if not os.path.exists(self.data_dir):
            url = root.yalefaces.loader.url
            self.warning("%s does not exist, downloading from %s...",
                         self.data_dir, url)
            dir_zip = wget.download("%s" % url, self.common_dir)
            try:
                with zipfile.ZipFile(dir_zip) as zip_file:
                    zip_file.extractall(self.common_dir)
            finally:
                os.remove(dir_zip)
        self.info("Loading from original Yale Faces files...")
        super(YaleFacesLoader, self).load_data()
        if self.shuffled_indices.mem is None:
            self.extract_validation_from_train(
                ratio=root.yalefaces.loader.validation_ratio)

    def get_label_from_filename(self, filename):
        dn = os.path.dirname(filename)
        return int(dn[-2:]) - 1

    def is_valid_filename(self, filename):
        return (os.path.splitext(filename)[1] == ".pgm"
                and filename.find("Ambient") < 0)


class YaleFacesWorkflow(StandardWorkflow):
    """
    Model was created for face recognition. Database - Yale Faces.
    Model - fully-connected Neural Network with SoftMax loss function.
    """
    def link_loader(self, init_unit):
        self.loader = YaleFacesLoader(
            self, on_device=root.yalefaces.loader.on_device,
            minibatch_size=root.yalefaces.loader.minibatch_size,
            train_paths=[root.yalefaces.loader.data_dir],
            data_dir=root.yalefaces.loader.data_dir,
            common_dir=root.yalefaces.loader.common_dir)
        self.loader.link_from(init_unit)


def run(load, main):
    load(YaleFacesWorkflow,
         fail_iterations=root.yalefaces.decision.fail_iterations,
         max_epochs=root.yalefaces.decision.max_epochs,
         prefix=root.yalefaces.snapshotter.prefix,
         snapshot_dir=root.common.snapshot_dir,
         layers=root.yalefaces.layers,
         loss_function=root.yalefaces.loss_function)
    main()
