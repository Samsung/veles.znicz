"""
Created on Jul 17, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""

import os


from veles.config import root
from veles.opencl_units import OpenCLWorkflow
from veles.snapshotter import Snapshotter
from veles.znicz.nn_units import Forward
from veles.mean_disp_normalizer import MeanDispNormalizer
from .forward_loader import ImagenetForwardLoader
from veles.znicz.tests.research.imagenet.forward_bbox import ImagenetBboxMapper


root.defaults = {
    "loader": {"year": "1",
               "series": "img"},
    "mapper": {"path": "",
               },
    "trained_workflow": "",
}

root.imagenet.from_snapshot_add_layer = False
IMAGENET_BASE_PATH = root.imagenet_base
root.loader.matrixes_filename = os.path.join(
    IMAGENET_BASE_PATH, "matrixes_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
root.loader.labels_filename = os.path.join(
    IMAGENET_BASE_PATH, "labels_int_%s_%s_0.txt" %
    (root.loader.year, root.loader.series))
root.loader.images_meta_filename = os.path.join(
    IMAGENET_BASE_PATH, "images_imagenet_%s_%s_%s_0.json" %
    (root.loader.year, root.loader.series, '%s'))


class ImagenetForward(OpenCLWorkflow):
    def __init__(self, workflow, **kwargs):
        super(ImagenetForward, self).__init__(workflow, **kwargs)
        self.train_wf = Snapshotter.import_(root.trained_workflow)
        units_to_remove = []
        for unit in self.train_wf:
            if not isinstance(unit, Forward) and \
               not isinstance(unit, MeanDispNormalizer):
                units_to_remove.append(unit)
        for unit in units_to_remove:
            self.train_wf.del_ref(unit)

        self.loader = ImagenetForwardLoader(
            self, root.loader.images_meta_filename,
            root.loader.labels_filename, root.loader.matrixes_filename)
        self.train_wf.link_from(self.loader)
        self.train_wf.fwds[0].link_attrs(self.loader, "minibatch_data")
        self.mapper = ImagenetBboxMapper(self, root.mapper.path)
        self.mapper.link_from(self.train_wf.fwds[-1])
        self.mapper.link_attrs(self.train_wf.fwds[-1],
                               ("output", "classified"))


def run(load, main):
    load(ImagenetForward, layers=root.imagenet.layers)
    main()
