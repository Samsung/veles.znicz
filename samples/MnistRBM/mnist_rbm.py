#!/usr/bin/python3 -O
"""
Created on Nov 20, 20134

Model created for digits recognition. Database - MNIST.
Model - RBM Neural Network.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import numpy
import scipy.io
from zope.interface import implementer

from veles.config import root
from veles.interaction import Shell
from veles.znicz.decision import TrivialDecision

import veles.znicz.loader as loader
import veles.znicz.nn_units as nn_units
import veles.znicz.rbm as RBM_units


@implementer(loader.IFullBatchLoader)
class MnistRBMLoader(loader.FullBatchLoader):
    def __init__(self, workflow, **kwargs):
        super(MnistRBMLoader, self).__init__(workflow, **kwargs)
        self.data_path = kwargs.get("data_path", "")

    def shuffle(self):
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
        self.train_indx = numpy.zeros((3, 1), dtype=numpy.int32)
        self.original_labels.mem = numpy.zeros([6400], dtype=numpy.int32)
        self.original_data.mem = numpy.zeros([6400, 196],
                                             dtype=numpy.float32)
        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = 4400
        init_data_path = self.data_path
        init_data = scipy.io.loadmat(init_data_path)

        self.original_data.mem[:] = init_data["patches"][:]


class MnistRBMWorkflow(nn_units.NNWorkflow):
    """
    Model created for digits recognition. Database - MNIST.
    Model - RBM Neural Network.
    """
    def __init__(self, workflow, layers, **kwargs):
        super(MnistRBMWorkflow, self).__init__(workflow, **kwargs)
        self.repeater.link_from(self.start_point)

        # LOADER
        self.loader = MnistRBMLoader(
            self, name="Mnist RBM fullbatch loader",
            minibatch_size=root.mnist_rbm.loader.minibatch_size,
            on_device=root.mnist_rbm.loader.on_device,
            data_path=root.mnist_rbm.loader.data_path)
        self.loader.link_from(self.repeater)

        # FORWARD UNIT
        all2all_rbm = RBM_units.All2AllRBM(
            self, output_shape=root.mnist_rbm.all2all.output_shape,
            weights_stddev=root.mnist_rbm.all2all.weights_stddev)
        self.forwards.append(all2all_rbm)
        self.forwards[0].link_from(self.loader)
        self.forwards[0].link_attrs(
            self.loader, ("input", "minibatch_data"),
            ("batch_size", "minibatch_size"))
        self.forwards[0].load_weights = True

        # EVALUATOR
        self.evaluator = RBM_units.EvaluatorRBM(self)
        self.evaluator.link_from(self.forwards[-1])
        self.evaluator.link_attrs(
            self.forwards[-1], ("input", "output"),
            ("output", "input"), ("ground_truth", "input"),
            "vbias", "bias", "weights")
        self.evaluator.link_attrs(
            self.loader, ("labels", "minibatch_labels"),
            ("batch_size", "minibatch_size"),
            ("max_samples_per_epoch", "total_samples"),
            "minibatch_class", "class_lengths",
            "last_minibatch", "max_minibatch_size")

        # DECISION
        self.decision = TrivialDecision(
            self, fail_iterations=root.mnist_rbm.decision.fail_iterations,
            max_epochs=root.mnist_rbm.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(
            self.loader, "minibatch_class", "minibatch_size",
            "last_minibatch", "class_lengths", "epoch_ended", "epoch_number")

        # link attribute max_epochs from self.decision to self.evaluator
        self.evaluator.link_attrs(self.decision, "max_epochs")

        # INTERPRETER PYTHON
        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        # GRADIENT
        del self.gds[:]
        gd_unit = RBM_units.GradientDescentRBM(
            self, learning_rate=root.mnist_rbm.learning_rate)
        self.gds.append(gd_unit)
        self.gds[0].link_from(self.ipython)
        self.gds[0].link_attrs(self.forwards[0], ("err_output", "output"))
        self.gds[0].link_attrs(
            self.forwards[-1], ("output", "output"),
            ("input", "input"), "weights", "bias", "vbias")
        self.gds[-1].gate_skip = self.decision.gate_skip
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].need_err_input = False
        self.repeater.link_from(self.gds[0])
        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        numpy.random.seed(1337)
        return super(MnistRBMWorkflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            device=device)


def run(load, main):
    load(MnistRBMWorkflow)
    main(learning_rate=root.mnist_rbm.learning_rate,
         weights_decay=root.mnist_rbm.weights_decay)
