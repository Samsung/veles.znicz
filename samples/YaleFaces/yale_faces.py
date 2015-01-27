#!/usr/bin/python3 -O
'''
Created on Nov 13, 2014

Model was created for face recognition. Database - Yale Faces.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
'''

from veles.config import root
from veles.znicz.downloader import Downloader
from veles.znicz.loader.image import FullBatchAutoLabelFileImageLoader
from veles.znicz.standard_workflow import StandardWorkflow


class YaleFacesWorkflow(StandardWorkflow):
    """
    Model was created for face recognition. Database - Yale Faces.
    Model - fully-connected Neural Network with SoftMax loss function.
    """

    def link_loader(self, init_unit):
        self.loader = FullBatchAutoLabelFileImageLoader(
            self, **root.yalefaces.loader.__dict__
        )
        self.loader.link_from(init_unit)

    def link_downloader(self, init_unit):
        self.downloader = Downloader(
            self,
            url="http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale."
                "zip",
            directory=root.common.test_dataset_root,
            files=["CroppedYale"])
        self.downloader.link_from(init_unit)

    def create_workflow(self):
        self.link_downloader(self.start_point)

        self.link_repeater(self.downloader)

        self.link_loader(self.repeater)

        self.link_forwards(self.loader, ("input", "minibatch_data"))

        self.link_evaluator(self.forwards[-1])

        self.link_decision(self.evaluator)

        self.link_snapshotter(self.decision)

        self.link_gds(self.snapshotter)

        self.link_end_point(self.gds[0])


def run(load, main):
    load(YaleFacesWorkflow,
         decision_config=root.yalefaces.decision,
         snapshotter_config=root.yalefaces.snapshotter,
         layers=root.yalefaces.layers,
         loss_function=root.yalefaces.loss_function)
    main()
