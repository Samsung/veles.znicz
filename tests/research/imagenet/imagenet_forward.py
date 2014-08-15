"""
Created on Jul 17, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""

import gc
import numpy
import os
import shutil
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
    merge_bboxes_by_dict, postprocess_bboxes_of_the_same_label


root.defaults = {
    "loader": {"year": "216_pool",
               "series": "img",
               "path": "/data/veles/datasets/imagenet",
               "path_to_bboxes":
               "/data/veles/datasets/imagenet/raw_bboxes/"
               # "raw_bboxes_4classes_img_val.4.pickle",
               "raw_bboxes_4classes_img_val.2.t_0_9.median.pickle",
               # "/data/veles/tmp/result_0_0_216_pool_img_test_1.json",
               "min_index": 0,
               "max_index": 0,
               "angle_step_final": numpy.pi / 12,
               "max_angle_final": numpy.pi / 12,
               "min_angle_final": (-numpy.pi / 12),
               "angle_step_merge": 1,
               "max_angle_merge": 0,
               "min_angle_merge": 0,
               "minibatch_size": 64,
               "only_this_file": "",
               "raw_bboxes_min_area": 256,
               "raw_bboxes_min_size": 8,
               "raw_bboxes_min_area_ratio": 0.005,
               "raw_bboxes_min_size_ratio": 0.05},
    "trained_workflow": "/data/veles/datasets/imagenet/snapshots/216_pool/"
                        "imagenet_ae_216_pool_27.12pt.3.pickle",
    "imagenet_base": "/data/veles/datasets/imagenet/temp",
    "result_path": "/data/veles/tmp/result_%d_%d_%s_%s_test_1.json",
    "mergebboxes": {"raw_path":
                    "/data/veles/tmp/result_raw_%d_%d_%s_%s_1.%d.pickle",
                    "ignore_negative": False,
                    "max_per_class": 6,
                    "probability_threshold": 0.98,
                    "last_chance_probability_threshold": 0.85,
                    "mode": "",
                    "labels_compatibility":
                    '/data/veles/datasets/imagenet/temp/216_pool/'
                    'label_compatibility.4.pickle',
                    "use_compatibility": True}
}

root.result_path = root.result_path % (
    root.loader.min_index, root.loader.max_index,
    root.loader.year, root.loader.series)


@implementer(IUnit)
class MergeBboxes(Unit):
    def __init__(self, workflow, labels_compatibility, **kwargs):
        super(MergeBboxes, self).__init__(workflow, **kwargs)
        self.winners = []
        self._image = ""
        self._image_size = 0
        self._bboxes = {}
        self.max_per_class = kwargs.get("max_per_class", 5)
        self.ignore_negative = kwargs.get("ignore_negative", True)
        self.save_raw = kwargs.get("save_raw_file_name", "")
        self.probability_threshold = kwargs.get("probability_threshold", 0.8)
        self.last_chance_probability_threshold = kwargs.get(
            "last_chance_probability_threshold", 0.7)
        self.rawfd = None
        self.labels_compatibility_file_name = labels_compatibility
        self.use_compatibility = kwargs.get("use_compatibility", True)
        self.demand("probabilities", "minibatch_bboxes", "minibatch_images",
                    "minibatch_size", "ended", "mode", "labels_mapping")

    def initialize(self, device, **kwargs):
        if self.save_raw:
            self.rawfd = open(self.save_raw, "wb")
        with open(self.labels_compatibility_file_name, 'rb') as fin:
            self.labels_compatibility, labels_array = pickle.load(fin)
        self.labels_compatibility_reverse_mapping = {
            lbl: index for index, lbl in enumerate(labels_array)}
        self.compatibility_threshold = numpy.mean(self.labels_compatibility)

    def validate(self, winners, image, raw):
        for winner in winners:
            bbox = winner[2]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                self.error("%s: validation failed\n\nwinners:\n%s\n\nraw:\n%s",
                           image, winners, raw.keys())

    def run(self):
        self.probabilities.map_read()

        if not self._image:
            self._image, self._image_size = self.minibatch_images[0]

        for index in range(self.minibatch_size):
            img, img_size = self.minibatch_images[index]
            if self._image != img:
                self._merge()
                self._image, self._image_size = img, img_size
            self.add_bbox(index, self.minibatch_bboxes[index])
        if self.ended and len(self._bboxes) > 0:
            self._merge()

    def reset(self):
        self._image = ""
        self.winners.clear()

    def add_bbox(self, index, bbox):
        key = tuple(bbox[0][k] for k in ("x", "y", "width", "height"))
        offset = 0 if self.ignore_negative else 1
        currbb = self._bboxes.get(key)
        if currbb is None:
            currbb = self.probabilities[index].copy()
        else:
            currbb[offset:] = numpy.maximum(
                currbb[offset:], self.probabilities[index][offset:])
            if not self.ignore_negative:
                currbb[0] = min(currbb[0], self.probabilities[index][0])
        self._bboxes[key] = currbb

    def _bboxes_compatibility(self, bbox1, bbox2):
        if bbox1[0] == bbox2[0]:
            return 1
        try:
            n1, n2 = (1 + self.labels_compatibility_reverse_mapping[
                self.labels_mapping[bbox[0] +
                                    (1 if self.ignore_negative else 0)]]
                      for bbox in (bbox1, bbox2))
        except KeyError:
            return 0
        return self.labels_compatibility[n1, n2]

    def _distance_from_center(self, bbox):
        dx = bbox[2][0] - self._image_size[1] / 2
        dy = bbox[2][1] - self._image_size[0] / 2
        return numpy.sqrt(dx * dx + dy * dy)

    def _remove_incompatible_bboxes(self, bboxes):
        sorted_bboxes = list(sorted(
            bboxes, key=lambda bbox: self._distance_from_center(bbox)))
        best = sorted_bboxes[0]
        winners = []
        for bbox in sorted_bboxes:
            if self._bboxes_compatibility(best, bbox) > \
               self.compatibility_threshold or \
               bbox[2][2] * bbox[2][3] >= best[2][2] * best[2][3]:
                winners.append(bbox)
        return winners

    def _merge(self):
        if self.rawfd is not None and self.mode == "merge":
            pickle.dump({self._image: self._bboxes},
                        self.rawfd, protocol=best_protocol)
            self.rawfd.flush()
            if self.ended:
                self.rawfd.close()
                self.info("Wrote %s", self.save_raw)
        if self.ignore_negative:
            for key in self._bboxes:
                self._bboxes[key] = self._bboxes[key][1:]
        if self.mode == "merge":
            winning_bboxes = merge_bboxes_by_dict(
                self._bboxes, pic_size=self._image_size)
            self.validate(winning_bboxes, self._image, self._bboxes)
            if not self.ignore_negative:
                tmp_bboxes = []
                for bbox in winning_bboxes[:self.max_per_class]:
                    if bbox[0] > 0:
                        tmp_bboxes.append(bbox)
                if len(tmp_bboxes) == 0:
                    for bbox in winning_bboxes[self.max_per_class:]:
                        if bbox[0] > 0:
                            tmp_bboxes.append(bbox)
                            if len(tmp_bboxes) >= self.max_per_class:
                                break
                winning_bboxes = tmp_bboxes
            self.debug("Merged %d bboxes of %s to %d bboxes",
                       len(self._bboxes), self._image, len(winning_bboxes))
        elif self.mode == "final":
            candidate_bboxes = []

            for bbox, probs in sorted(self._bboxes.items()):
                self.debug("%s: %s %s", self._image, bbox, probs)
                maxidx = numpy.argmax(probs)
                if not self.ignore_negative and maxidx == 0:
                    continue
                prob = probs[maxidx]
                if prob >= self.last_chance_probability_threshold:
                    candidate_bboxes.append((maxidx, prob, bbox))
            candidate_bboxes = postprocess_bboxes_of_the_same_label(
                candidate_bboxes)

            self.debug("Picked %d candidates", len(candidate_bboxes))
            winning_bboxes = []
            max_prob = 0
            max_prob_bbox = None
            for bbox in candidate_bboxes:
                self.debug("%s: %s %d %.3f", self._image, bbox[2], *bbox[:2])
                prob = bbox[1]
                if prob > max_prob:
                    max_prob = prob
                    max_prob_bbox = bbox
                if prob >= self.probability_threshold:
                    winning_bboxes.append(bbox)
            if len(winning_bboxes) == 0 and max_prob_bbox is not None:
                winning_bboxes.append(max_prob_bbox)
                self.debug("%s: used last chance, %s", self._image,
                           max_prob_bbox)
            if self.use_compatibility and len(winning_bboxes) > 1:
                winning_bboxes = self._remove_incompatible_bboxes(
                    winning_bboxes)
            self.debug("%d bboxes win", len(winning_bboxes))
        else:
            assert False
        if len(winning_bboxes) > 0:
            self.winners.append({"path": self._image, "bbxs": winning_bboxes})
        self._bboxes = {}


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
            min_index=root.loader.min_index,
            max_index=root.loader.max_index,
            angle_step=root.loader.angle_step_merge,
            max_angle=root.loader.max_angle_merge,
            min_angle=root.loader.min_angle_merge,
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

        lc_probability_threshold = \
            root.mergebboxes.last_chance_probability_threshold
        self.mergebboxes = MergeBboxes(
            self, save_raw_file_name=root.mergebboxes.raw_path % (
                root.loader.min_index, root.loader.max_index, root.loader.year,
                root.loader.series, best_protocol),
            ignore_negative=root.mergebboxes.ignore_negative,
            max_per_class=root.mergebboxes.max_per_class,
            probability_threshold=root.mergebboxes.probability_threshold,
            last_chance_probability_threshold=lc_probability_threshold,
            labels_compatibility=root.mergebboxes.labels_compatibility,
            use_compatibility=root.mergebboxes.use_compatibility)
        self.mergebboxes.link_attrs(self.fwds[-1],
                                    ("probabilities", "output"))
        self.mergebboxes.link_attrs(self.loader, "ended", "minibatch_bboxes",
                                    "minibatch_size", "minibatch_images")
        self.json_writer = ImagenetResultWriter(
            self, root.loader.labels_int_dir, root.result_path,
            ignore_negative=root.mergebboxes.ignore_negative)
        self.mergebboxes.link_attrs(self.json_writer, "labels_mapping")
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

    def on_workflow_finished(self, force_propagate=False):
        if self.loader.mode == "merge":
            print('-' * 80)
            print('====  FINAL  ===')
            print('-' * 80)
            self.loader.angle_step = root.loader.angle_step_final
            self.loader.min_angle = root.loader.min_angle_final
            self.loader.max_angle = root.loader.max_angle_final
            self.loader.bboxes_file_name = root.result_path
            shutil.copy(root.result_path, root.result_path + ".raw")
            self.loader.reset()
            if self.loader.total == 0:
                print("No bboxes for final stage - return")
                super(ImagenetForward, self).on_workflow_finished()
                return
            self.mergebboxes.reset()
            self.run()
        else:
            super(ImagenetForward, self).on_workflow_finished()


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
