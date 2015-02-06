#!/usr/bin/python3 -O
# encoding: utf-8
"""
Created on June 29, 2013


Model created for Chinese characters recognition. Dataset was generated by
VELES with generate_kanji.py utility.
Model – fully-connected Neural Network with MSE loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
import os
import pickle
import re

import numpy
from zope.interface import implementer

from veles.config import root
import veles.error as error
import veles.memory as formats
import veles.opencl_types as opencl_types
import veles.prng as rnd
import veles.loader as loader
from veles.znicz.standard_workflow import StandardWorkflow


@implementer(loader.IFullBatchLoader)
class KanjiLoader(loader.FullBatchLoaderMSE):
    """Loads dataset.
    """

    def __init__(self, workflow, **kwargs):
        self.train_path = kwargs["train_path"]
        self.target_path = kwargs["target_path"]
        super(KanjiLoader, self).__init__(workflow, **kwargs)
        self.class_targets = formats.Vector()

    def __getstate__(self):
        state = super(KanjiLoader, self).__getstate__()
        state["index_map"] = None
        return state

    def load_data(self):
        """Load the data here.

        Should be filled here:
            class_lengths[].
        """
        fin = open(root.kanji.index_map, "rb")
        self.index_map = pickle.load(fin)
        fin.close()

        fin = open(os.path.join(self.train_path, self.index_map[0]), "rb")
        self.first_sample = pickle.load(fin)["data"]
        fin.close()

        fin = open(self.target_path, "rb")
        targets = pickle.load(fin)
        fin.close()
        self.class_targets.reset()
        sh = [len(targets)]
        sh.extend(targets[0].shape)
        self.class_targets.mem = numpy.empty(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])
        for i, target in enumerate(targets):
            self.class_targets[i] = target

        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = len(self.index_map)

        self.original_labels.mem = numpy.empty(
            len(self.index_map), dtype=numpy.int32)
        lbl_re = re.compile("^(\d+)_\d+/(\d+)\.\d\.pickle$")
        for i, fnme in enumerate(self.index_map):
            res = lbl_re.search(fnme)
            if res is None:
                raise error.BadFormatError("Incorrectly formatted filename "
                                           "found: %s" % (fnme))
            lbl = int(res.group(1))
            self.original_labels[i] = lbl
            idx = int(res.group(2))
            if idx != i:
                raise error.BadFormatError("Incorrect sample index extracted "
                                           "from filename: %s " % (fnme))

        self.info("Found %d samples. Extracting 15%% for validation..." % (
            len(self.index_map)))
        self.resize_validation(rand=rnd.get(2), ratio=0.15)
        self.info("Extracted, resulting datasets are: [%s]" % (
            ", ".join(str(x) for x in self.class_lengths)))

    def create_minibatch_data(self):
        """Allocate arrays for minibatch_data etc. here.
        """
        sh = [self.max_minibatch_size]
        sh.extend(self.first_sample.shape)
        self.minibatch_data.mem = numpy.zeros(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])

        sh = [self.max_minibatch_size]
        sh.extend((self.class_targets[0].size,))
        self.minibatch_targets.mem = numpy.zeros(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])

    def fill_minibatch(self):
        """Fill minibatch data labels and indexes according to current shuffle.
        """
        idxs = self.minibatch_indices.mem
        for i, ii in enumerate(idxs[:self.minibatch_size]):
            fnme = "%s/%s" % (self.train_path, self.index_map[ii])
            fin = open(fnme, "rb")
            sample = pickle.load(fin)
            data = sample["data"]
            lbl = sample["lbl"]
            fin.close()
            self.minibatch_data[i] = data
            self.minibatch_labels[i] = lbl
            self.minibatch_targets[i] = self.class_targets[lbl].reshape(
                self.minibatch_targets[i].shape)


class KanjiWorkflow(StandardWorkflow):
    """
    Model created for Chinese characters recognition. Dataset was generated by
    VELES with generate_kanji.py utility.
    Model – fully-connected Neural Network with MSE loss function.
    """
    def create_workflow(self):
        # Add repeater unit
        self.link_repeater(self.start_point)

        # Add loader unit
        self.link_loader(self.repeater)

        # Add fwds units
        self.link_forwards(self.loader, ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.link_evaluator(self.forwards[-1])

        # Add decision unit
        self.link_decision(self.evaluator)

        # Add snapshotter unit
        self.link_snapshotter(self.decision)

        # Add gradient descent units
        self.link_gds(self.snapshotter)

        if root.kanji.add_plotters:
            # Add plotters units
            self.link_error_plotter(self.gds[0])
            self.link_weights_plotter(
                self.error_plotter[-1], layers=root.kanji.layers,
                limit=root.kanji.weights_plotter.limit,
                weights_input="weights")
            self.link_max_plotter(self.weights_plotter[-1])
            self.link_min_plotter(self.max_plotter[-1])
            self.link_mse_plotter(self.min_plotter[-1])
            last = self.mse_plotter[-1]
        else:
            last = self.gds[0]

        # Add end point unit
        self.link_end_point(last)

    def initialize(self, device, weights, bias, **kwargs):
        super(KanjiWorkflow, self).initialize(device=device)
        if weights is not None:
            for i, fwds in enumerate(self.forwards):
                fwds.weights.map_invalidate()
                fwds.weights.mem[:] = weights[i][:]
        if bias is not None:
            for i, fwds in enumerate(self.forwards):
                fwds.bias.map_invalidate()
                fwds.bias.mem[:] = bias[i][:]

    def link_loader(self, init_unit):
        self.loader = KanjiLoader(
            self, train_path=root.kanji.data_paths.train,
            target_path=root.kanji.data_paths.target,
            **root.kanji.loader.__dict__)
        self.loader.link_from(init_unit)


def run(load, main):
    weights = None
    bias = None
    w, snapshot = load(
        KanjiWorkflow,
        decision_config=root.kanji.decision,
        snapshotter_config=root.kanji.snapshotter,
        layers=root.kanji.layers,
        loss_function=root.kanji.loss_function)
    if snapshot:
        if type(w) == tuple:
            logging.info("Will load weights")
            weights = w[0]
            bias = w[1]
        else:
            logging.info("Will load workflow")
            logging.info("Weights and bias ranges per layer are:")
            for fwds in w.fwds:
                logging.info("%f %f %f %f" % (
                    fwds.weights.mem.min(), fwds.weights.mem.max(),
                    fwds.bias.mem.min(), fwds.bias.mem.max()))
            w.decision.improved <<= True
    main(weights=weights, bias=bias)
