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
            self, on_device=root.yalefaces.loader.on_device,
            minibatch_size=root.yalefaces.loader.minibatch_size,
            train_paths=root.yalefaces.loader.train_paths,
            filename_types=root.yalefaces.loader.filename_types,
            ignored_files=root.yalefaces.loader.ignored_files,
            validation_ratio=root.yalefaces.loader.validation_ratio,
            shuffle_limit=root.yalefaces.loader.shuffle_limit,
            add_sobel=root.yalefaces.loader.add_sobel,
            normalization_type=root.yalefaces.loader.normalization_type,
            mirror=root.yalefaces.loader.mirror,
            color_space=root.yalefaces.loader.color_space,
            background_color=root.yalefaces.loader.background_color
            )
        self.loader.link_from(init_unit)
    # TODO(lyubov.p): add url unit before loader when Url Unit exists
    # "url": "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip"


def run(load, main):
    load(YaleFacesWorkflow,
         fail_iterations=root.yalefaces.decision.fail_iterations,
         max_epochs=root.yalefaces.decision.max_epochs,
         prefix=root.yalefaces.snapshotter.prefix,
         snapshot_dir=root.common.snapshot_dir,
         layers=root.yalefaces.layers,
         loss_function=root.yalefaces.loss_function,)
    main()
