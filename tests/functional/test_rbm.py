#!/usr/bin/python3 -O
"""
Created on November 6, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

import numpy
import os
import scipy.io
from zope.interface import implementer

from veles.config import root
from veles.interaction import Shell
from veles.tests import multi_device
from veles.znicz.decision import TrivialDecision
import veles.loader as loader
import veles.prng as prng
import veles.znicz.nn_units as nn_units
import veles.znicz.rbm_units as rbm_units
from veles.znicz.tests.functional import StandardTest


root.mnist_rbm.update({
    "all2all": {"weights_stddev": 0.05, "output_sample_shape": 1000},
    "decision": {"max_epochs": 2},
    "snapshotter": {"prefix": "mnist_rbm"},
    "loader": {"minibatch_size": 128, "force_cpu": True,
               "data_path":
               os.path.join(os.path.dirname(__file__), "..",
                            'data', 'rbm_data', 'test_rbm_functional.mat')},
    "learning_rate": 0.03,
    "weights_decay": 0.0005,
    "factor_ortho": 0.0})


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
            self.train_indx[cur_class] += 1
            if self.train_indx[cur_class] == self.class_lengths[cur_class]:
                self.train_indx[cur_class] = 0

    def load_data(self):
        self.train_indx = numpy.zeros((3, 1), dtype=numpy.int32)
        self.original_data.mem = numpy.zeros([1000, 196],
                                             dtype=numpy.float32)
        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = 240
        init_data_path = self.data_path
        init_data = scipy.io.loadmat(init_data_path)

        self.original_data.mem[:] = init_data["patches"][:]


class MnistRBMWorkflow(nn_units.NNWorkflow):
    """
    Model created for digits recognition. Database - MNIST.
    Model - RBM Neural Network.
    """
    def __init__(self, workflow, **kwargs):
        super(MnistRBMWorkflow, self).__init__(workflow, **kwargs)
        self.repeater.link_from(self.start_point)
        # LOADER
        self.loader = MnistRBMLoader(
            self, name="Mnist RBM fullbatch loader",
            minibatch_size=root.mnist_rbm.loader.minibatch_size,
            force_cpu=root.mnist_rbm.loader.force_cpu,
            data_path=root.mnist_rbm.loader.data_path)
        self.loader.link_from(self.repeater)

        # FORWARD UNIT
        b1 = rbm_units.Binarization(self)
        del self.forwards[:]
        self.forwards.append(b1)
        self.forwards[0].link_from(self.loader)
        self.forwards[0].link_attrs(
            self.loader, ("input", "minibatch_data"),
                         ("batch_size", "minibatch_size"))
        # EVALUATOR
        a2a = rbm_units.All2AllSigmoid(
            self,
            output_sample_shape=root.mnist_rbm.all2all.output_sample_shape,
            weights_stddev=root.mnist_rbm.all2all.weights_stddev)
        self.forwards.append(a2a)
        self.forwards[1].link_from(self.forwards[0])
        self.forwards[1].link_attrs(self.forwards[0], ("input", "output"))
        self.evaluator = rbm_units.EvaluatorRBM(self, bias_shape=196)
        self.evaluator.link_from(self.forwards[1])
        self.evaluator.link_attrs(self.forwards[1], "weights")
        self.evaluator.link_attrs(self.forwards[1], ("input", "output"))
        self.evaluator.link_attrs(self.forwards[0], ("target", "output"))
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"))

        # DECISION
        self.decision = TrivialDecision(
            self, max_epochs=root.mnist_rbm.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(
            self.loader, "minibatch_class", "minibatch_size",
            "last_minibatch", "class_lengths", "epoch_ended", "epoch_number")

        # INTERPRETER PYTHON
        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        # GRADIENT
        del self.gds[:]
        gd_unit = rbm_units.GradientRBM(self, stddev=0.05, v_size=196,
                                        h_size=1000, cd_k=1)
        self.gds.append(gd_unit)
        gd_unit.link_from(self.ipython)
        gd_unit.link_attrs(self.forwards[1], ("input", "output"),
                           ("hbias", "bias"), "weights")
        gd_unit.link_attrs(self.loader, ("batch_size", "minibatch_size"))
        gd_unit.link_attrs(self.evaluator, "vbias")
        gd_unit = rbm_units.BatchWeights(self)
        gd_unit.name = "BatchWeights #1"
        self.gds.append(gd_unit)
        gd_unit.link_from(self.gds[0])
        gd_unit.link_attrs(self.forwards[0], ("v", "output"))
        gd_unit.link_attrs(self.forwards[1], ("h", "output"))
        # Question minibatsh_size is current size of batch
        gd_unit.link_attrs(self.loader, ("batch_size", "minibatch_size"))
        gd_unit = rbm_units.BatchWeights2(self)
        gd_unit.name = "BatchWeights #2"
        self.gds.append(gd_unit)
        gd_unit.link_from(self.gds[1])
        gd_unit.link_attrs(self.gds[0], ("v", "v1"), ("h", "h1"))
        # Question minibatsh_size is current size of batch
        gd_unit.link_attrs(self.loader, ("batch_size", "minibatch_size"))
        gd_unit = rbm_units.GradientsCalculator(self)
        self.gds.append(gd_unit)
        gd_unit.link_from(self.gds[2])
        gd_unit.link_attrs(self.gds[1], ("hbias0", "hbias_batch"),
                           ("vbias0", "vbias_batch"),
                           ("weights0", "weights_batch"))
        gd_unit.link_attrs(self.gds[2], ("hbias1", "hbias_batch"),
                           ("vbias1", "vbias_batch"),
                           ("weights1", "weights_batch"))
        gd_unit = rbm_units.WeightsUpdater(self, learning_rate=0.001)
        self.gds.append(gd_unit)
        gd_unit.link_from(self.gds[3])
        gd_unit.link_attrs(self.gds[3], "hbias_grad", "vbias_grad",
                           "weights_grad")
        gd_unit.link_attrs(self.forwards[1], ("weights", "weights"),
                           ("hbias", "bias"))
        gd_unit.link_attrs(self.evaluator, "vbias")
        self.repeater.link_from(gd_unit)
        self.end_point.link_from(gd_unit)
        self.end_point.gate_block = ~self.decision.complete

        self.repeater.gate_block = self.decision.complete

    def initialize(self, device, learning_rate, weights_decay, snapshot=False,
                   **kwargs):
        return super(MnistRBMWorkflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            snapshot=False, device=device)


class TestRBMworkflow(StandardTest):
    """Test RBM workflow for MNIST.
    """
    @multi_device()
    def test_rbm(self):
        """This function creates RBM workflow for MNIST task
        and compares result with the output produced function RBM
        from  MATLAB (http://deeplearning.net/tutorial/rbm.html (25.11.14))
        Raises:
            AssertLess: if unit output is wrong.
        """
        init_weights = scipy.io.loadmat(os.path.join(
            os.path.dirname(__file__), '..', 'data', 'rbm_data',
            'R_141014_init.mat'))
        learned_weights = scipy.io.loadmat(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'rbm_data',
                         'R_141014_learned.mat'))
        self.info("MNIST RBM TEST")
        workflow = MnistRBMWorkflow(self.parent)
        workflow.initialize(device=self.device, learning_rate=0,
                            weights_decay=0)
        workflow.forwards[1].weights.map_write()
        workflow.forwards[1].bias.map_write()
        workflow.evaluator.vbias.map_write()
        workflow.forwards[1].weights.mem[:] = init_weights["W"].transpose()[:]
        workflow.forwards[1].bias.mem[:] = init_weights["hbias"].ravel()[:]
        workflow.evaluator.vbias.mem[:] = init_weights["vbias"].ravel()[:]
        prng.get().seed(1337)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        workflow.gds[4].weights.map_read()
        workflow.gds[4].hbias.map_read()
        workflow.gds[4].vbias.map_read()
        diffW = numpy.sum(numpy.abs(learned_weights["W"] -
                          workflow.gds[4].weights.mem.transpose()))
        diffHbias = numpy.sum(numpy.abs(learned_weights["hbias"].ravel() -
                              workflow.gds[4].hbias.mem.ravel()))
        diffVbias = numpy.sum(numpy.abs(learned_weights["vbias"].ravel() -
                              workflow.gds[4].vbias.mem.ravel()))

        self.assertLess(diffW, 1e-12, " diff with learned weights is %0.17f"
                        % diffW)

        self.assertLess(diffVbias, 1e-12,
                        " diff with learned vbias is %0.17f" % diffVbias)
        self.assertLess(diffHbias, 1e-12,
                        " diff with learned hbias is %0.17f" % diffHbias)


if __name__ == "__main__":
    StandardTest.main()
