#!/usr/bin/python3 -O
"""
Created on November 18, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
import unittest

import numpy

import veles.memory as formats
from veles.dummy import DummyWorkflow
from veles.tests import AcceleratedTest, assign_backend
import veles.znicz.kohonen as kohonen


class TestKohonen(AcceleratedTest):
    ABSTRACT = True

    def setUp(self):
        super(TestKohonen, self).setUp()
        self.input = numpy.array([[1, 2, 3, 2, 1],
                                  [0, 1, 2, 1, 0],
                                  [0, 1, 0, 1, 0],
                                  [2, 0, 1, 0, 2],
                                  [1, 0, 1, 0, 1]],
                                 dtype=self.dtype)
        self.weights = numpy.array([[1, 0, 2, 1, -1],
                                    [3, 1, 0, 2, 3],
                                    [-1, 2, 0, 1, 3],
                                    [0, 1, -1, 0, 1],
                                    [-1, -1, 1, 1, 1],
                                    [1, -2, -1, -1, 3],
                                    [-1, -2, 1, 3, 1],
                                    [-1, -1, 3, 0, 2],
                                    [1, 0, 3, 2, -1]],
                                   dtype=self.dtype)
        self.output = numpy.array([8, 0, 3, 1, 0], dtype=self.dtype)
        self.total = numpy.append(self.output, self.output)
        self.new_weights = numpy.array(
            [[0.0095, 3.4077, -1.1363, -0.2331, 7.291],
             [-7.3005, -0.141, 6.3435, -3.8349, -7.3005],
             [7.0286, -3.361, 6.0806, 0.0389, -6.5709],
             [3.7339, -0.0242, 10.1774, 3.7097, 0.],
             [7.3563, 7.3657, 2.8212, 0.121, 0.1115],
             [0.2757, 9.8744, 9.1679, 6.6905, -6.0922],
             [6.6781, 10.1743, 2.6081, -6.4829, 0.0152],
             [6.8938, 7.1, -4.0215, 3.6939, -3.3245],
             [0.213, 3.6071, -3.4613, -2.7033, 6.5233]],
            dtype=self.dtype)
        self.winners = numpy.array([2, 1, 0, 1, 0, 0, 0, 0, 1],
                                   dtype=numpy.int)

    def test_forward(self):
        self.info("Will test KohonenForward unit forward pass")
        c = kohonen.KohonenForward(DummyWorkflow())
        c.input = formats.Vector()
        c.input.mem = self.input[:]
        c.weights = formats.Vector()
        c.weights.mem = self.weights[:]
        c.initialize(device=self.device)

        c.cpu_run()
        max_diff = numpy.fabs(self.output.ravel() - c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)

        c.ocl_run()
        c.output.map_read()  # get results back
        max_diff = numpy.fabs(self.output.ravel() - c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)

    def test_forward_total(self):
        c = kohonen.KohonenForward(DummyWorkflow(), total=True)
        c.input = formats.Vector()
        c.input.mem = self.input[:]
        c.weights = formats.Vector()
        c.weights.mem = self.weights[:]
        c.minibatch_size = 5
        c.minibatch_offset = 5
        c.batch_size = 10
        c.initialize(device=self.device)

        c.cpu_run()
        c.minibatch_offset = 10
        c.cpu_run()
        max_diff = numpy.fabs(self.total.ravel() - c.total.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)
        c.total.map_invalidate()
        c.total.mem[:] = 0

        c.minibatch_offset = 5
        c.ocl_run()
        c.minibatch_offset = 10
        c.ocl_run()
        c.total.map_read()  # get results back
        max_diff = numpy.fabs(self.total.ravel() - c.total.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)

    def test_forward_with_argmins(self):
        self.info("Will test KohonenForward unit forward pass")
        c = kohonen.KohonenForward(DummyWorkflow())
        c.input = formats.Vector()
        c.input.mem = self.input[:]
        c.weights = formats.Vector()
        c.weights.mem = self.weights[:]
        c.argmins = formats.Vector()
        c.argmins.mem = self.output[:]
        c.initialize(device=self.device)
        c.argmins.initialize(self.device)

        c.cpu_run()
        max_diff = numpy.fabs(self.output.ravel() - c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)
        c.output.map_invalidate()
        c.output.mem[:] = 0

        c.ocl_run()
        c.output.map_read()  # get results back
        max_diff = numpy.fabs(self.output.ravel() - c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)

    def test_forward_total_with_argmins(self):
        c = kohonen.KohonenForward(DummyWorkflow(), total=True)
        c.input = formats.Vector()
        c.input.mem = self.input[:]
        c.weights = formats.Vector()
        c.weights.mem = self.weights[:]
        c.argmins = formats.Vector()
        c.argmins.mem = self.output[:]
        c.minibatch_size = 5
        c.minibatch_offset = 5
        c.batch_size = 10
        c.initialize(device=self.device)
        c.argmins.initialize(self.device)

        c.cpu_run()
        c.minibatch_offset = 10
        c.cpu_run()
        max_diff = numpy.fabs(self.total.ravel() - c.total.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)
        c.total.map_invalidate()
        c.total.mem[:] = 0

        c.minibatch_offset = 5
        c.ocl_run()
        c.minibatch_offset = 10
        c.ocl_run()
        c.total.map_read()  # get results back
        max_diff = numpy.fabs(self.total.ravel() - c.total.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)

    def test_train(self):
        self.info("Will test KohonenForward unit train pass")

        c = kohonen.KohonenTrainer(DummyWorkflow(), shape=(3, 3))
        c.input = formats.Vector()
        c.input.mem = self.input[:]
        c.gradient_decay = lambda t: 1.0 / (1.0 + t)
        c.weights.mem = self.weights[:]

        c.initialize(device=self.device)

        c.cpu_run()

        weights = c.weights.mem.ravel()
        winners = c.winners.mem
        max_diff = numpy.fabs(self.new_weights.ravel() - weights.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)
        self.assertTrue(all(winners == self.winners),
                        "Wrong winners %s" % str(winners))
        self.assertEqual(1, c.time)

        c.weights.map_invalidate()
        c.weights.mem[:] = self.weights
        c.winners.map_invalidate()
        c.winners.mem[:] = 0
        c.unmap_vectors(c.weights, c.winners)
        c.time = 0

        c.ocl_run()

        c.weights.map_read()
        c.winners.map_read()
        weights = c.weights.mem.ravel()
        winners = c.winners.mem

        max_diff = numpy.fabs(self.new_weights.ravel() - weights.ravel()).max()
        self.assertLess(max_diff, 0.0001, "Result differs by %.5f" % max_diff)
        self.assertTrue(all(winners == self.winners),
                        "Wrong winners %s" % str(winners))

    def test_train_2d(self):
        c = kohonen.KohonenTrainer(DummyWorkflow(), shape=(8, 8))
        c.input = formats.Vector()
        c.input.mem = numpy.array([[0.31550583, 1.1270231],
                                   [0.29612723, -0.06573299],
                                   [1.31921649, 1.05750644],
                                   [0.08247627, 1.02469325],
                                   [1.16866457, 0.17982987],
                                   [1.02037144, 1.01223791],
                                   [0.98056835, -0.07386222],
                                   [1.38947701, 0.92844582],
                                   [-0.19097605, 0.93884891],
                                   [-0.0462381, 0.9390552]],
                                  dtype=self.dtype)
        c.weights.mem = numpy.array([[-0.0016954, -0.04789865],
                                     [-0.02529763, -0.01292748],
                                     [0.03493399, -0.0210943],
                                     [0.04232741, -0.04312364],
                                     [-0.02253316, -0.0356375],
                                     [-0.02234498, -0.04476132],
                                     [-0.04966503, -0.01858753],
                                     [-0.00078752, -0.02389915],
                                     [-0.02921833, 0.02203111],
                                     [-0.01250004, -0.0007029],
                                     [-0.0392132, 0.04369496],
                                     [0.0227923, 0.00068291],
                                     [-0.03300898, 0.02275362],
                                     [-0.04002197, 0.01962677],
                                     [0.04527853, 0.01763646],
                                     [0.02422675, -0.0014066],
                                     [0.02988973, -0.00767365],
                                     [0.02501957, 0.0270541],
                                     [-0.04602004, -0.04211192],
                                     [-0.00852699, -0.0050553],
                                     [-0.04696839, -0.03786743],
                                     [0.02532622, -0.01868231],
                                     [-0.04766771, -0.03533152],
                                     [0.04065625, 0.03299204],
                                     [0.02376441, -0.02513403],
                                     [-0.04940482, -0.0361849],
                                     [0.04200317, -0.00837324],
                                     [-0.034668, 0.02090169],
                                     [0.03051241, -0.00888058],
                                     [-0.01763904, -0.03941356],
                                     [-0.04220719, -0.00880075],
                                     [0.00380691, 0.00257401],
                                     [-0.01285645, -0.00317304],
                                     [0.03942059, 0.01659596],
                                     [0.02748787, 0.03347495],
                                     [0.0174959, 0.04006275],
                                     [0.00401112, 0.02466389],
                                     [-0.03625129, 0.04108988],
                                     [-0.04164252, 0.01556505],
                                     [0.01967308, -0.00163736],
                                     [0.00229704, 0.03010981],
                                     [-0.0486289, 0.04920848],
                                     [0.00499213, 0.04573606],
                                     [-0.02147039, -0.04091481],
                                     [0.00268905, 0.01209371],
                                     [0.02065182, -0.04725563],
                                     [0.03775178, 0.04930231],
                                     [0.01848668, 0.0230999],
                                     [0.02313485, -0.01656069],
                                     [-0.02108706, 0.01281377],
                                     [-0.04907014, -0.00975056],
                                     [0.0005148, 0.006321],
                                     [-0.01666364, -0.0324413],
                                     [0.01713218, 0.04932376],
                                     [0.00060229, 0.01657825],
                                     [0.03282689, -0.01097561],
                                     [-0.00745258, 0.03435414],
                                     [0.0287052, 0.0411209],
                                     [-0.03609686, -0.02597349],
                                     [-0.00829501, 0.01966193],
                                     [-0.03073766, -0.00672367],
                                     [-0.02448536, 0.04296542],
                                     [-0.04278564, 0.02849008],
                                     [-0.03142332, 0.01513203]],
                                    dtype=self.dtype)
        weights_copy = numpy.copy(c.weights.mem)
        c.initialize(device=self.device)
        c.cpu_run()
        valid_winners = numpy.copy(c.winners.mem)
        self.assertEqual(10, sum(valid_winners))
        valid_weights = numpy.copy(c.weights.mem)
        c.weights.map_invalidate()
        c.weights.mem[:] = weights_copy
        c.winners.map_invalidate()
        c.winners.mem[:] = 0
        c.time = 0
        c.ocl_run()
        c.winners.map_read()
        self.assertTrue(all(c.winners.mem == valid_winners))
        c.weights.map_read()
        max_diff = numpy.fabs(
            valid_weights.ravel() - c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))


@assign_backend("ocl")
class OpenCLTestKohonen(TestKohonen):
    pass


@assign_backend("cuda")
@unittest.expectedFailure
class CUDATestKohonen(TestKohonen):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
