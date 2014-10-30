#!/usr/bin/python3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import numpy
import os
import scipy.io
from zope.interface import implementer

from veles.config import root
from veles.interaction import Shell
from veles.znicz.decision import TrivialDecision

import veles.znicz.loader as loader
import veles.znicz.nn_units as nn_units
import veles.znicz.rbm as RBM_units


root.mnist.update({
    "all2all": {"weights_stddev": 0.05},
    "decision": {"fail_iterations": 100,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "mnist"},
    "loader": {"minibatch_size": 128, "on_device": True},
    "learning_rate": 0.03,
    "weights_decay": 0.0005,  # 1.6%
    "factor_ortho": 0.0})


@implementer(loader.IFullBatchLoader)
class MnistLoader(loader.FullBatchLoader):
    """Loads MNIST dataset.
    """
    def shuffle(self):
        """Randomly shuffles the TRAIN dataset.
        """
        if self.shuffled_indices.mem is None:
            self.shuffled_indices.mem = numpy.arange(self.total_samples,
                                                     dtype=numpy.int32)
        self.debug("Shuffled TRAIN")

    def fill_minibatch(self):
        idxs = self.minibatch_indices.mem

        cur_class = self._minibatch_class
        for i, ii in enumerate(idxs[:self.minibatch_size]):
            self.minibatch_data[i] = \
                self.original_data[self.train_indx[cur_class]]
            self.train_indx[cur_class] = self.train_indx[cur_class] + 1
            if self.train_indx[cur_class] == self.class_lengths[cur_class]:
                self.train_indx[cur_class] = 0
        if self.original_labels:
            for i, ii in enumerate(idxs[:self.minibatch_size]):
                self.minibatch_labels[i] = self.original_labels[int(ii)]

    def load_data(self):
        """Here we load MNIST data.
        """
        self.train_indx = numpy.zeros((3, 1), dtype=numpy.int32)
        self.original_labels.mem = numpy.zeros([6400], dtype=numpy.int32)
        self.original_data.mem = numpy.zeros([6400, 196],
                                             dtype=numpy.float32)
        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = 4400
        init_data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'tests', 'unit', 'data',
            'rbm', 'test_rbm.mat'))
        init_data = scipy.io.loadmat(init_data_path)

        self.original_data.mem[:] = init_data["patches"][:]


class MnistRBMWorkflow(nn_units.NNWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    """
    def __init__(self, workflow, layers, **kwargs):
        super(MnistRBMWorkflow, self).__init__(workflow, **kwargs)
        self.repeater.link_from(self.start_point)
# LOADER
        self.loader = MnistLoader(
            self, name="Mnist fullbatch loader",
            minibatch_size=128,
            on_device=root.mnist.loader.on_device)
        self.loader.link_from(self.repeater)

# FORWARD UNIT
        aa = RBM_units.All2AllRBM(
            self, output_shape=1000,
            weights_stddev=root.mnist.all2all.weights_stddev)
        self.fwds.append(aa)
        self.fwds[0].link_from(self.loader)
        self.fwds[0].link_attrs(self.loader,
                                ("input", "minibatch_data"),
                                ("batch_size", "minibatch_size"))
        self.fwds[0].load_weights = True
# EVALUATOR
        self.evaluator = RBM_units.EvaluatorRBM(self)
        self.evaluator.link_from(self.fwds[0])
        self.evaluator.link_attrs(self.fwds[0],
                                  ("input", "output"),
                                  ("output", "input"),
                                  ("ground_truth", "input"),
                                  "vbias", "bias", "weights")
        self.evaluator.link_attrs(self.loader, ("labels", "minibatch_labels"),
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  "minibatch_class", "class_lengths",
                                  "last_minibatch", "max_minibatch_size")
# DECISION
        self.decision = TrivialDecision(
            self, fail_iterations=root.mnist.decision.fail_iterations,
            max_epochs=2)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class", "minibatch_size",
                                 "last_minibatch", "class_lengths",
                                 "epoch_ended", "epoch_number")
# link attribute max_epochs from self.decision to self.evaluator
        self.evaluator.link_attrs(self.decision, "max_epochs")
# INTERPRETER PYTHON
        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

# GRADIENT
        del self.gds[:]
        gd_unit = RBM_units.GradientDescentRBM(
            self, learning_rate=root.mnist.learning_rate)
        self.gds.append(gd_unit)
        self.gds[0].link_from(self.ipython)
        self.gds[0].link_attrs(self.fwds[0], ("err_output", "output"))
        self.gds[0].link_attrs(self.fwds[-1],
                               ("output", "output"),
                               ("input", "input"),
                               "weights", "bias", "vbias")
        self.gds[-1].gate_skip = self.decision.gate_skip
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].need_err_input = False
        self.repeater.link_from(self.gds[0])
        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        a = super(MnistRBMWorkflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            device=device)
        numpy.random.seed(1337)
        return a


def run(load, main):
    load(MnistRBMWorkflow)
    main(learning_rate=root.mnist.learning_rate,
         weights_decay=root.mnist.weights_decay)
