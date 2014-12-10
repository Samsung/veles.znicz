#!/usr/bin/python3 -O
"""
Created on Nov 14, 2014

Model created for object recognition. Database - Fake Imagenet (1000 classes
of random pictures 227*227). Self-constructing Model. It means that Model can
change for any Model (Convolutional, Fully connected, different parameters) in
configuration file.

Copyright (c) 2014 Samsung R&D Institute Russia
"""

import json
import numpy
import os
import pickle
from zope.interface import implementer

from veles.config import root
import veles.error as error
from veles.memory import Vector
import veles.opencl_types as opencl_types
import veles.znicz.loader as loader
from veles.znicz.standard_workflow import StandardWorkflow

data_dir = "/data/veles/datasets/FakeImagenet/Veles"
root.imagenet.loader.names_labels_filename = os.path.join(
    data_dir, "fake_original_labels.pickle")
root.imagenet.loader.count_samples_filename = os.path.join(
    data_dir, "fake_count_samples.json")
root.imagenet.loader.samples_filename = os.path.join(
    data_dir, "fake_original_data.dat")
root.imagenet.loader.matrixes_filename = os.path.join(
    data_dir, "fake_matrixes.pickle")


@implementer(loader.ILoader)
class ImagenetLoader(loader.Loader):
    """loads imagenet from samples.dat, labels.pickle"""
    def __init__(self, workflow, **kwargs):
        super(ImagenetLoader, self).__init__(workflow, **kwargs)
        self.mean = Vector()
        self.rdisp = Vector()
        self.file_samples = ""
        self.sx = root.imagenet.loader.sx
        self.sy = root.imagenet.loader.sy

    def init_unpickled(self):
        super(ImagenetLoader, self).init_unpickled()
        self.original_labels = None

    def __getstate__(self):
        stt = super(ImagenetLoader, self).__getstate__()
        stt["original_labels"] = None
        stt["file_samples"] = None
        return stt

    def load_data(self):
        self.original_labels = []

        with open(root.imagenet.loader.names_labels_filename, "rb") as fin:
            for lbl in pickle.load(fin):
                self.original_labels.append(int(lbl))
        self.info("Labels (min max count): %d %d %d",
                  numpy.min(self.original_labels),
                  numpy.max(self.original_labels),
                  len(self.original_labels))

        with open(root.imagenet.loader.count_samples_filename, "r") as fin:
            for i, n in enumerate(json.load(fin)):
                self.class_lengths[i] = n
        self.info("Class Lengths: %s", str(self.class_lengths))

        if numpy.sum(self.class_lengths) != len(self.original_labels):
            raise error.Bug(
                "Number of labels missmatches sum of class lengths")

        with open(root.imagenet.loader.matrixes_filename, "rb") as fin:
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

        self.file_samples = open(root.imagenet.loader.samples_filename,
                                 "rb")
        if (self.file_samples.seek(0, 2) // (self.sx * self.sy * 3) !=
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
        idxs[:count] = self.shuffled_indices[start_offset:start_offset + count]

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


class ImagenetWorkflow(StandardWorkflow):
    """
    Model created for object recognition. Database - Fake Imagenet (1000
    classes of random pictures 227*227). Self-constructing Model. It means that
    Model can change for any Model (Convolutional, Fully connected, different
    parameters) in configuration file.
    """
    def __init__(self, workflow, **kwargs):
        super(ImagenetWorkflow, self).__init__(
            workflow,
            fail_iterations=root.imagenet.decision.fail_iterations,
            max_epochs=root.imagenet.decision.max_epochs,
            prefix=root.imagenet.snapshotter.prefix,
            snapshot_interval=root.imagenet.snapshotter.interval,
            snapshot_dir=root.common.snapshot_dir,
            layers=root.imagenet.layers,
            loss_function=root.imagenet.loss_function, ** kwargs)

    def create_workflow(self):
        # Add repeater unit
        self.link_repeater(self.start_point)

        # Add loader unit
        self.link_loader(self.repeater)

        # Add meandispnormalizer unit
        self.link_meandispnorm(self.loader)

        # Add fwds units
        self.link_forwards(self.meandispnorm, ("input", "output"))

        # Add evaluator for single minibatch
        self.link_evaluator(self.forwards[-1])

        # Add decision unit
        self.link_decision(self.evaluator)

        # Add snapshotter unit
        self.link_snapshotter(self.decision)

        # Add gradient descent units
        self.link_gds(self.snapshotter)

        if root.imagenet.add_plotters:
            # Add error plotter unit
            self.link_error_plotter(self.gds[0])

            # Add Confusion matrix plotter unit
            self.link_conf_matrix_plotter(self.error_plotter[-1])

            # Add Err y plotter unit
            self.link_err_y_plotter(self.conf_matrix_plotter[-1])

            # Add Weights plotter unit
            self.link_weights_plotter(
                self.err_y_plotter[-1], layers=root.imagenet.layers,
                limit=root.imagenet.weights_plotter.limit,
                weights_input="weights")

            last = self.weights_plotter[-1]
        else:
            last = self.gds[0]

        # Add end_point unit
        self.link_end_point(last)

    def link_loader(self, init_unit):
        self.loader = ImagenetLoader(
            self, on_device=root.imagenet.loader.on_device,
            minibatch_size=root.imagenet.loader.minibatch_size,
            shuffle_limit=root.imagenet.loader.shuffle_limit)
        self.loader.link_from(init_unit)


def run(load, main):
    load(ImagenetWorkflow)
    main()
