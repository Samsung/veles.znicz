#!/usr/bin/python3 -O
'''
Created on Nov 13, 2014

Model was created for face recognition. Database - Yale Faces.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
'''


from veles.config import root
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
    # TODO(lyubov.p): add url unit before loader when Url Unit exists
    # "url": "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip"


def run(load, main):
    load(YaleFacesWorkflow,
         decision_config=root.yalefaces.decision,
         snapshotter_config=root.yalefaces.snapshotter,
         layers=root.yalefaces.layers,
         loss_function=root.yalefaces.loss_function)
    main()
