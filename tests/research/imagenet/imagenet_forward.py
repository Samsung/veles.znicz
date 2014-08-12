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
from veles.pickle2 import pickle, best_protocol
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
from veles.distributable import IDistributable


root.defaults = {
    "loader": {"year": "216_pool",
               "series": "img",
               "path": "/data/veles/datasets/imagenet",
               "path_to_bboxes":
               "/data/veles/datasets/imagenet/raw_bboxes/"
               "raw_bboxes_4classes_img_val.4.pickle",
               # "/data/veles/tmp/result_216_pool_img_test_0.json",
               # "/data/veles/tmp/result_216_pool_img_test_1.json",
               "angle_step": 0.01,
               "max_angle": 0,
               "min_angle": 0,
               "minibatch_size": 64,
               "only_this_file": "00007197",
               "raw_bboxes_min_area": 256,
               "raw_bboxes_min_size": 8,
               "raw_bboxes_min_area_ratio": 0.005,
               "raw_bboxes_min_size_ratio": 0.05},
    "trained_workflow": "/data/veles/datasets/imagenet/snapshots/216_pool/"
                        "imagenet_ae_216_pool_27.12pt.3.pickle",
    "imagenet_base": "/data/veles/datasets/imagenet/temp",
    "result_path": "/data/veles/tmp/result_%s_%s_test_0.json",
    "mergebboxes": {"raw_path":
                    "/data/veles/tmp/result_raw_%s_%s_0.%d.pickle",
                    "ignore_negative": False,
                    "max_per_class": 5,
                    "probability_threshold": 0.8,
                    "mode": ""}
}

root.result_path = root.result_path % (root.loader.year, root.loader.series)


@implementer(IUnit, IDistributable)
class MergeBboxes(Unit):
    def __init__(self, workflow, **kwargs):
        super(MergeBboxes, self).__init__(workflow, **kwargs)
        self.winners = []
        self._prev_image = ""
        self._prev_image_size = 0
        self._current_bboxes = {}
        self.max_per_class = kwargs.get("max_per_class", 5)
        self.ignore_negative = kwargs.get("ignore_negative", True)
        self.save_raw = kwargs.get("save_raw_file_name", "")
        self.probability_threshold = kwargs.get("probability_threshold", 0.8)
        self.rawfd = None
        self.demand("probabilities", "current_image", "minibatch_bboxes",
                    "minibatch_size", "ended", "current_image_size", "mode")

    def initialize(self, device, **kwargs):
        if self.save_raw:
            self.rawfd = open(self.save_raw, "wb")

    def validate(self, winners, image, raw):
        for winner in winners:
            bbox = winner[2]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                self.error("%s: validation failed\n\nwinners:\n%s\n\nraw:\n%s",
                           image, winners, raw.keys())

    def run(self):
        self.probabilities.map_read()
        if not self._prev_image:
            self._prev_image_size = self.current_image_size
            self._prev_image = self.current_image

        for index, bbox in enumerate(
                self.minibatch_bboxes[:self.minibatch_size]):
            key = tuple(bbox[0][k] for k in ("x", "y", "width", "height"))
            self._current_bboxes[key] = numpy.maximum(
                self._current_bboxes[key]
                if key in self._current_bboxes else 0,
                self.probabilities[index])
            offset = 0 if self.ignore_negative else 1
            currbb = self._current_bboxes.get(key)
            if currbb is None:
                currbb = self.probabilities[index]
            else:
                currbb[offset:] = numpy.maximum(
                    currbb[offset:], self.probabilities[index][offset:])
                if not self.ignore_negative:
                    currbb[0] = min(currbb[0], self.probabilities[index][0])
            self._current_bboxes[key] = currbb

        if self.current_image != self._prev_image or self.ended:
            prev_image = self._prev_image
            self._prev_image = self.current_image
            if self.rawfd is not None:
                pickle.dump({prev_image: self._current_bboxes},
                            self.rawfd, protocol=best_protocol)
                self.rawfd.flush()
                if self.ended:
                    self.rawfd.close()
                    self.info("Wrote %s", self.save_raw)
            if self.ignore_negative:
                for key in self._current_bboxes:
                    self._current_bboxes[key] = self._current_bboxes[key][1:]
            if self.mode == "merge":
                self.debug("Merging %d bboxes of %s",
                           len(self._current_bboxes), prev_image)
                winning_bboxes = merge_bboxes_by_dict(
                    self._current_bboxes, self._prev_image_size,
                    self.max_per_class)
                self.validate(winning_bboxes, prev_image, self._current_bboxes)
            elif self.mode == "final":
                winning_bboxes = []
                for bbox, probs in sorted(self._current_bboxes.items()):
                    print(bbox, probs)
                    maxidx = numpy.argmax(probs)
                    if not self.ignore_negative and maxidx == 0:
                        continue
                    prob = probs[maxidx]
                    if prob >= self.probability_threshold:
                        winning_bboxes.append((maxidx, prob, bbox))
            else:
                assert False
            self.winners.append({"path": prev_image, "bbxs": winning_bboxes})
            self._current_bboxes = {}
            self._prev_image_size = self.current_image_size

    def apply_data_from_slave(self, data, slave):
        pass

    def drop_slave(self, slave):
        pass

    def apply_data_from_master(self, data):
        self._prev_image = self.current_image
        self._current_bboxes.clear()

    def generate_data_for_master(self):
        return None

    def generate_data_for_slave(self, slave):
        return {}


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
            max_angle=root.loader.max_angle,
            min_angle=root.loader.min_angle,
            only_this_file=root.loader.only_this_file,
            raw_bboxes_min_area=root.loader.raw_bboxes_min_area,
            raw_bboxes_min_size=root.loader.raw_bboxes_min_size,
            raw_bboxes_min_area_ratio=root.loader.raw_bboxes_min_area_ratio,
            raw_bboxes_min_size_ratio=root.loader.raw_bboxes_min_size_ratio)
        self.loader.link_from(self.repeater)
        self.loader.gate_block = self.loader.ended

        self.fwds = []
        for fwd in train_wf.fwds:
            fwd.workflow = self
            self.fwds.append(fwd)
        del train_wf.fwds[:]

        self.loader.entry_shape = list(self.fwds[0].input.shape)
        self.loader.entry_shape[0] = root.loader.minibatch_size
        self.meandispnorm = train_wf.meandispnorm
        self.meandispnorm.workflow = self
        self.meandispnorm.link_from(self.loader)
        self.meandispnorm.link_attrs(self.loader, ("input", "minibatch_data"))
        self.loader.link_attrs(self.meandispnorm, "mean")
        self.fwds[0].link_from(self.meandispnorm)
        self.fwds[0].link_attrs(self.meandispnorm, ("input", "output"))

        self.mergebboxes = MergeBboxes(
            self, save_raw_file_name=root.mergebboxes.raw_path % (
                root.loader.year, root.loader.series, best_protocol),
            ignore_negative=root.mergebboxes.ignore_negative,
            max_per_class=root.mergebboxes.max_per_class,
            probability_threshold=root.mergebboxes.probability_threshold)
        self.mergebboxes.link_attrs(self.fwds[-1],
                                    ("probabilities", "output"))
        self.mergebboxes.link_attrs(self.loader, "current_image", "ended",
                                    "minibatch_bboxes", "minibatch_size",
                                    "current_image_size")
        self.json_writer = ImagenetResultWriter(
            self, root.loader.labels_int_dir, root.result_path,
            ignore_negative=root.mergebboxes.ignore_negative)
        if root.mergebboxes.mode:
            self.mergebboxes.mode = root.mergebboxes.mode
            self.json_writer.mode = root.mergebboxes.mode
        else:
            self.mergebboxes.link_attrs(self.loader, "mode")
            self.json_writer.link_attrs(self.loader, "mode")
        self.mergebboxes.link_from(self.fwds[-1])
        self.repeater.link_from(self.mergebboxes)

        self.json_writer.link_attrs(self.mergebboxes, "winners")
        self.json_writer.link_from(self.mergebboxes)
        self.end_point.link_from(self.json_writer)
        self.json_writer.gate_block = ~self.loader.ended


def run(load, main):
    numpy.set_printoptions(precision=3, suppress=True)
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
    main(forward_mode=True, minibatch_size=root.loader.minibatch_size)
