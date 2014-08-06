"""
Created on Jul 17, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""

import gc
import numpy
import os
import sys
from zope.interface import implementer


from veles.config import root
from veles.formats import Vector
from veles.mean_disp_normalizer import MeanDispNormalizer
from veles.opencl_units import OpenCLWorkflow
from veles.snapshotter import Snapshotter
from veles.mutable import Bool
from veles.znicz.nn_units import Forward
from veles.units import Unit, IUnit
from veles.workflow import Repeater
from veles.znicz.tests.research.imagenet.forward_loader import \
    ImagenetForwardLoaderBbox


root.defaults = {
    "loader": {"year": "216_pool",
               "series": "img",
               "do_shuffle": False,
               "path": "/data/veles/datasets/imagenet",
               "path_to_bboxes": "/data/veles/datasets/imagenet/raw_bboxes/"
                                 "raw_bboxes_4classes_img_val.4.pickle",
               "angle_step": numpy.pi / 6,
               "max_angle": numpy.pi},
    "trained_workflow": "/data/veles/datasets/imagenet/snapshots/216_pool/"
                        "imagenet_ae_216_pool_27.12pt.3.pickle",
    "imagenet_base": "/data/veles/datasets/imagenet/temp"
}


@implementer(IUnit)
class MergeBboxes(Unit):
    def __init__(self, workflow, **kwargs):
        super(MergeBboxes, self).__init__(workflow, **kwargs)
        self.probabilities = None
        self.path_to_bboxes = kwargs.get("path_to_bboxes", "")
        self.bbox_for_merge = Vector()
        self.aperture = None
        self.channels = None
        self.bboxes_for_merge = []
        self.image_end = Bool(False)
        self.ind_minibatch_end = 0
        self.ind_minibatch_begin = 0

    def initialize(self, device, **kwargs):
        self.bboxes_for_merge = []

    def run(self):
        if self.image_end:
            # merge
            del self.bboxes_for_merge[:]
            self.image_end = False
        if self.image_end is not True:
            count_bboxes_in_minibatch = self.probabilities.shape[0]
            self.ind_minibatch_end += count_bboxes_in_minibatch
            for i in range(self.ind_minibatch_begin, self.ind_minibatch_end):
                self.bboxes_for_merge.append(self.probabilities[i])
            self.ind_minibatch_begin += count_bboxes_in_minibatch


class ImagenetForward(OpenCLWorkflow):
    def __init__(self, workflow, **kwargs):
        super(ImagenetForward, self).__init__(workflow, **kwargs)
        sys.path.append(os.path.dirname(__file__))
        self.info("Importing %s...", root.trained_workflow)
        train_wf = Snapshotter.import_(root.trained_workflow)
        self.info("Loaded workflow %s" % train_wf)
        gc.collect()
        units_to_remove = []
        for unit in train_wf:
            if (not isinstance(unit, Forward) and
                    not isinstance(unit, MeanDispNormalizer)):
                units_to_remove.append(unit)
        self.info("units_to_remove %s" % units_to_remove)
        for unit in units_to_remove:
            unit.unlink_all()
            train_wf.del_ref(unit)

        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)

        self.loader = ImagenetForwardLoaderBbox(
            self,
            bboxes_file_name=root.loader.path_to_bboxes,
            angle_step=root.loader.angle_step,
            max_angle=root.loader.max_angle)
        self.loader.link_from(self.repeater)

        self.fwds = []
        for fwd in train_wf.fwds:
            fwd.workflow = self
            self.fwds.append(fwd)
        del train_wf.fwds[:]

        self.loader.link_attrs(self.fwds[0], ("entry", "input"))
        self.meandispnorm = train_wf.meandispnorm
        self.meandispnorm.workflow = self
        self.meandispnorm.link_from(self.loader)
        self.loader.link_attrs(self.meandispnorm, "mean")
        self.fwds[0].link_from(self.meandispnorm)
        self.fwds[0].link_attrs(self.loader, "minibatch_data")
        self.repeater.link_from(self.fwds[-1])
        self.loader.gate_block = self.loader.ended

        """
        self.mergebboxes = MergeBboxes(
            self, path_to_bboxes=root.loader.path_to_bboxes)
        self.mergebboxes.link_attrs(self.fwds[-1],
                                    ("probabilities", "output"))
        self.mergebboxes.link_attrs(
            self.loader, "aperture", "channels")
        self.mergebboxes.link_from(self.fwds[-1])

        self.end_point.link_from(self.mergebboxes)
        """
        self.end_point.link_from(self.loader)
        self.end_point.gate_block = ~self.loader.ended


def run(load, main):
    root.imagenet.from_snapshot_add_layer = False
    IMAGENET_BASE_PATH = root.loader.path
    CACHED_DATA_FNME = os.path.join(IMAGENET_BASE_PATH, str(root.loader.year))
    root.loader.names_labels_filename = os.path.join(
        CACHED_DATA_FNME, "original_labels_%s_%s_0_forward.pickle" %
        (root.loader.year, root.loader.series))
    root.loader.samples_filename = os.path.join(
        CACHED_DATA_FNME, "original_data_%s_%s_0_forward.dat" %
        (root.loader.year, root.loader.series))
    root.loader.labels_int_dir = os.path.join(
        CACHED_DATA_FNME, "labels_int_%s_%s_0.txt" %
        (root.loader.year, root.loader.series))
    load(ImagenetForward)
    main()
