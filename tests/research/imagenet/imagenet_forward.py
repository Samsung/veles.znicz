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
from veles.mean_disp_normalizer import MeanDispNormalizer
from veles.opencl_units import OpenCLWorkflow
from veles.snapshotter import Snapshotter
from veles.znicz.nn_units import Forward
from veles.units import Unit, IUnit
from veles.workflow import Repeater
from veles.znicz.tests.research.imagenet.forward_loader import \
    ImagenetForwardLoaderBbox
from veles.znicz.tests.research.imagenet.forward_json import \
    ImagenetResultWriter
from veles.znicz.tests.research.imagenet.forward_bbox import \
    merge_bboxes_by_dict


root.defaults = {
    "loader": {"year": "216_pool",
               "series": "img",
               "do_shuffle": False,
               "path": "/data/veles/datasets/imagenet",
               "path_to_bboxes": "/data/veles/datasets/imagenet/raw_bboxes/"
                                 "raw_bboxes_4classes_img_val.4.pickle",
               "angle_step": numpy.pi / 4,
               "max_angle": numpy.pi},
    "trained_workflow": "/data/veles/datasets/imagenet/snapshots/216_pool/"
                        "imagenet_ae_216_pool_27.12pt.3.pickle",
    "imagenet_base": "/data/veles/datasets/imagenet/temp",
    "result_path": "/data/veles/tmp/result_%s_%s_0.json"
}

root.result_path = root.result_path % (root.loader.year, root.loader.series)


@implementer(IUnit)
class MergeBboxes(Unit):
    def __init__(self, workflow, **kwargs):
        super(MergeBboxes, self).__init__(workflow, **kwargs)
        self.winners = []
        self._prev_image = ""
        self._current_bboxes = {}
        self.max_per_class = kwargs.get("max_per_class", 5)
        self.demand("probabilities", "current_image", "minibatch_bboxes",
                    "minibatch_size", "ended", "current_image_size")

    def initialize(self, device, **kwargs):
        pass

    def run(self):
        self.probabilities.map_read()

        if self.current_image != self._prev_image or self.ended:
            prev_image = self._prev_image
            self._prev_image = self.current_image
            if prev_image:
                self.debug("Merging %d bboxes", len(self._current_bboxes))
                winning_bboxes = merge_bboxes_by_dict(
                    self._current_bboxes, self.current_image_size,
                    self.max_per_class)
                self.winners.append({"path": self.current_image,
                                     "bbxs": winning_bboxes})
                self._current_bboxes = {}
        for index, bbox in enumerate(
                self.minibatch_bboxes[:self.minibatch_size]):
            key = tuple(bbox[k] for k in ("x", "y", "width", "height"))
            self._current_bboxes[key] = numpy.maximum(
                self._current_bboxes[key]
                if key in self._current_bboxes else 0,
                self.probabilities[index])


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
        self.loader.gate_block = self.loader.ended

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

        self.mergebboxes = MergeBboxes(self)
        self.mergebboxes.link_attrs(self.fwds[-1],
                                    ("probabilities", "output"))
        self.mergebboxes.link_attrs(self.loader, "current_image", "ended",
                                    "minibatch_bboxes", "minibatch_size",
                                    "current_image_size")
        self.mergebboxes.link_from(self.fwds[-1])
        self.repeater.link_from(self.mergebboxes)

        self.json_writer = ImagenetResultWriter(
            self, root.loader.labels_int_dir, root.result_path)
        self.json_writer.link_attrs(self.mergebboxes, "winners")
        self.json_writer.link_from(self.mergebboxes)
        self.end_point.link_from(self.json_writer)
        self.json_writer.gate_block = ~self.loader.ended


def run(load, main):
    root.imagenet.from_snapshot_add_layer = False
    CACHED_DATA_FNME = os.path.join(root.imagenet_base, str(root.loader.year))
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
