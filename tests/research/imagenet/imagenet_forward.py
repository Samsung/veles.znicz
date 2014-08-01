"""
Created on Jul 17, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""

import numpy
import pickle
import os
import sys
from zope.interface import implementer


from veles.config import root
import veles.error as error
from veles.formats import Vector
from veles.mean_disp_normalizer import MeanDispNormalizer
import veles.opencl_types as opencl_types
from veles.opencl_units import OpenCLWorkflow
from veles.snapshotter import Snapshotter
import veles.znicz.loader as loader
from veles.mutable import Bool
from veles.znicz.nn_units import Forward
from veles.units import Unit, IUnit
from veles.znicz.tests.research.imagenet.forward_loader import \
    ImagenetForwardLoaderBbox


root.defaults = {
    "loader": {"year": "216_pool",
               "series": "img",
               "minibatch_size": 360,
               "do_shuffle": False,
               "path": "/data/veles/datasets/imagenet",
               "aperture": 216,
               "channels": 4},
    "trained_workflow": "/data/veles/datasets/imagenet/snapshots/"
                        "216_pool/imagenet_ae_216_pool_0.026746.3.pickle",
    "imagenet_base": "/data/veles/datasets/imagenet/temp"
}


@implementer(loader.ILoader)
class Loader(loader.Loader):
    """loads imagenet from samples.dat, labels.pickle"""
    def __init__(self, workflow, **kwargs):
        super(Loader, self).__init__(workflow, **kwargs)
        self.mean = Vector()
        self.rdisp = Vector()
        self.file_samples = ""
        self.sx = root.loader.sx
        self.sy = root.loader.sy
        self.channels = root.loader.channels
        self.do_shuffle = kwargs.get("do_shuffle", False)

    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self.original_labels = None

    def __getstate__(self):
        stt = super(Loader, self).__getstate__()
        stt["original_labels"] = None
        stt["file_samples"] = None
        return stt

    def load_data(self):
        self.original_labels = []

        with open(root.loader.names_labels_filename, "rb") as fin:
            for lbl in pickle.load(fin):
                self.original_labels.append(int(lbl))
        self.info("Labels (min max count): %d %d %d",
                  numpy.min(self.original_labels),
                  numpy.max(self.original_labels),
                  len(self.original_labels))

        self.class_lengths = [0, len(self.original_labels), 0]

        if numpy.sum(self.class_lengths) != len(self.original_labels):
            raise error.Bug(
                "Number of labels missmatches sum of class lengths")

        with open(root.loader.matrixes_filename, "rb") as fin:
            matrixes = pickle.load(fin)

        self.mean.mem = matrixes[0]
        self.rdisp.mem = matrixes[1].astype(
            opencl_types.dtypes[root.common.precision_type])
        if numpy.count_nonzero(numpy.isnan(self.rdisp.mem)):
            raise ValueError("rdisp matrix has NaNs")
        if numpy.count_nonzero(numpy.isinf(self.rdisp.mem)):
            raise ValueError("rdisp matrix has Infs")
        if self.mean.shape != self.rdisp.shape:
            raise ValueError("mean.shape != rdisp.shape")
        if self.mean.shape[0] != self.sy or self.mean.shape[1] != self.sx:
            raise ValueError("mean.shape != (%d, %d)" % (self.sy, self.sx))

        self.file_samples = open(root.loader.samples_filename, "rb")
        if (self.file_samples.seek(0, 2)
                // (self.sx * self.sy * self.channels) !=
                len(self.original_labels)):
            raise error.Bug("Wrong data file size")

    def create_minibatches(self):
        sh = [self.max_minibatch_size]
        sh.extend(self.mean.shape)
        self.minibatch_data.mem = numpy.zeros(sh, dtype=numpy.uint8)
        sh = [self.max_minibatch_size]
        self.minibatch_labels.mem = numpy.zeros(sh, dtype=numpy.int32)
        self.minibatch_indices.mem = numpy.zeros(self.max_minibatch_size,
                                                 dtype=numpy.int32)

    def fill_indices(self, start_offset, count):
        self.minibatch_indices.map_invalidate()
        idxs = self.minibatch_indices.mem
        self.shuffled_indices.map_read()
        self.info("count %s" % count)
        self.info("start_offset %s" % start_offset)
        self.info("self.shuffled_indices %s"
                  % self.shuffled_indices.mem)

        idxs[:count] = self.shuffled_indices[start_offset:start_offset + count]
        self.info("idxs[:count] %s" % idxs[:count])

        if self.is_master:
            return True

        self.minibatch_data.map_invalidate()
        self.minibatch_labels.map_invalidate()

        sample_bytes = self.mean.mem.nbytes

        for i, ii in enumerate(idxs[:count]):
            self.file_samples.seek(int(ii) * sample_bytes)
            self.file_samples.readinto(self.minibatch_data.mem[i])
            self.minibatch_labels.mem[i] = self.original_labels[int(ii)]

        if count < len(idxs):
            idxs[count:] = self.class_lengths[1]  # no data sample is there
            self.minibatch_data.mem[count:] = self.mean.mem
            self.minibatch_labels.mem[count:] = 0  # 0 is no data

        return True

    def fill_minibatch(self):
        raise error.Bug("Control should not go here")

    def shuffle(self):
        self.info("do_shuffle %s" % self.do_shuffle)
        if self.do_shuffle:
            super(Loader, self).shuffle()

    def _update_total_samples(self):
        """Fills self.class_offsets from self.class_lengths.
        """
        total_samples = 0
        for i, n in enumerate(self.class_lengths):
            total_samples += n
            self.class_offsets[i] = total_samples
        self.total_samples = total_samples


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
        train_wf = Snapshotter.import_(root.trained_workflow)
        self.info("self.train_wf %s" % train_wf)
        units_to_remove = []
        for unit in train_wf:
            if (not isinstance(unit, Forward) and
                    not isinstance(unit, MeanDispNormalizer)):
                units_to_remove.append(unit)
        self.info("units_to_remove %s" % units_to_remove)
        for unit in units_to_remove:
            unit.unlink_all()
            train_wf.del_ref(unit)

        self.loader = ImagenetForwardLoaderBbox(
            self,
            minibatch_size=root.loader.minibatch_size,
            bboxes_file_name=root.loader.path_to_bboxes,
            aperture=root.loader.aperture, channels=root.loader.channels,
            matrices_pickle=root.loader.matrixes_filename)
        self.loader.link_from(self.start_point)

        self.fwds = []
        for fwd in train_wf.fwds:
            fwd.workflow = self
            self.fwds.append(fwd)
        del train_wf.fwds[:]

        self.meandispnorm = train_wf.meandispnorm
        self.meandispnorm.workflow = self
        self.meandispnorm.link_from(self.loader)
        self.fwds[0].link_attrs(self.loader, "minibatch_data")

        self.mergebboxes = MergeBboxes(
            self, path_to_bboxes=root.loader.path_to_bboxes)
        self.mergebboxes.link_attrs(self.fwds[-1],
                                    ("probabilities", "output"))
        self.mergebboxes.link_attrs(
            self.loader, "aperture", "channels")
        self.mergebboxes.link_from(self.fwds[-1])

        self.end_point.link_from(self.mergebboxes)


def run(load, main):
    root.imagenet.from_snapshot_add_layer = False
    IMAGENET_BASE_PATH = root.loader.path
    CACHED_DATA_FNME = os.path.join(IMAGENET_BASE_PATH, str(root.loader.year))
    root.loader.path_to_bboxes = os.path.join(
        CACHED_DATA_FNME,
        "images_imagenet_%s_%s_validation_0_forward.4.pck4" %
        (root.loader.year, root.loader.series))
    root.loader.names_labels_filename = os.path.join(
        CACHED_DATA_FNME, "original_labels_%s_%s_0_forward.pickle" %
        (root.loader.year, root.loader.series))
    root.loader.samples_filename = os.path.join(
        CACHED_DATA_FNME, "original_data_%s_%s_0_forward.dat" %
        (root.loader.year, root.loader.series))
    root.loader.matrixes_filename = os.path.join(
        CACHED_DATA_FNME, "matrixes_%s_%s_0.pickle" %
        (root.loader.year, root.loader.series))
    root.loader.labels_int_dir = os.path.join(
        CACHED_DATA_FNME, "labels_int_%s_%s_0.txt" %
        (root.loader.year, root.loader.series))
    load(ImagenetForward)
    main()
