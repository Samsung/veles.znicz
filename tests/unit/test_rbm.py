# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Dec 4, 2014

Unit test for RBM.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import numpy
import os
import scipy.io
from veles.backends import NumpyDevice

from veles.dummy import DummyLauncher
from veles.memory import Array
import veles.prng as prng
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.all2all import All2AllSigmoid
import veles.znicz.rbm_units as rbm


class TestRBMUnits(AcceleratedTest):
    """Tests for uniits used in RBM workflow.
    """
    ABSTRACT = True

    def test_All2AllSigmoid(self):
        """This function creates All2AllRBM unit for MNIST task
        and compares result with the output produced function RBM
        from  MATLAB (http://deeplearning.net/tutorial/rbm.html (25.11.14))
        Raises:
            AssertLess: if unit output is wrong.
        """
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "research/MnistRBM/rbm_data",
            "test_a2a.mat")
        test_data = scipy.io.loadmat(data_path)
        a2a = All2AllSigmoid(self.parent, output_sample_shape=196,
                             weights_stddev=0.05)
        # add initialize and input
        a2a.input = Array()
        a2a.input.reset()
        a2a.input.mem = numpy.zeros((128, 1000), numpy.float64)
        a2a.input.mem[:] = test_data['hr'][:]
        a2a.weights_transposed = True
        a2a.batch_size = 128
        a2a.initialize(device=NumpyDevice())
        a2a.weights.map_write()
        a2a.bias.map_write()
        a2a.weights.reset()
        a2a.weights.mem = numpy.transpose(test_data["W"])[:]
        a2a.bias.mem = numpy.transpose(test_data["vbias"])[:]
        a2a.run()
        a2a.output.map_read()
        diff = numpy.sum(numpy.abs(a2a.output.mem - test_data['vr']))
        self.assertLess(diff, 1e-12, " total error  is %0.17f" % diff)

    def test_Binarization(self):
        b1 = rbm.Binarization(self.parent)
        # add initialize and input
        b1.input = Array()
        b1.input.reset()
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "research/MnistRBM/rbm_data",
            "test_bino.mat")
        test_data = scipy.io.loadmat(data_path)
        v1 = test_data["v1"]
        v1bino = test_data["v1bino"]
        b1.input.mem = v1
        b1.batch_size = v1bino.shape[0]
        prng.get().seed(1337)
        b1.initialize(device=NumpyDevice())
        b1.run()
        b1.output.map_read()
        diff = numpy.sum(numpy.abs(b1.output.mem - v1bino))
        self.assertLess(diff, 1e-12,
                        "total error weights is %0.17f" % diff)

    def test_EvaluatorRBM(self):
        evl = rbm.EvaluatorRBM(self.parent, bias_shape=196)
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "research/MnistRBM/rbm_data",
            "test_eval.mat")
        test_data = scipy.io.loadmat(data_path)
        evl.input = Array()
        evl.input.mem = test_data["h"]
        evl.weights = Array()
        evl.weights.mem = test_data["W"].transpose()
        evl.target = Array()
        evl.target.mem = test_data["ground_truth"]
        evl.batch_size = 128
        evl.initialize(device=NumpyDevice())
        evl.rec.bias.mem[:] = test_data["vbias"].ravel()[:]
        prng.get().seed(1337)
        evl.run()
        diff = numpy.sum(numpy.abs(evl.mse.mse.mem - test_data["err"].ravel()))
        self.assertLess(diff, 1e-12,
                        "total error weights is %0.17f" % diff)

    def test_GradientRBM(self):
        launcher = DummyLauncher()
        gds = rbm.GradientRBM(launcher, stddev=0.05, v_size=196,
                              h_size=1000, cd_k=1)
        gds.input = Array()
        gds.weights = Array()
        gds.vbias = Array()
        gds.hbias = Array()
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "research/MnistRBM/rbm_data",
            "test_grad.mat")
        grad_data = scipy.io.loadmat(data_path)
        gds.input.reset()
        gds.input.mem = numpy.zeros((128, 1000),
                                    dtype=numpy.float64)

        gds.input.map_write()
        gds.input.mem[:] = grad_data["h1_in"][:]
        gds.weights.mem = grad_data["W"][:].transpose()
        gds.hbias = Array()
        gds.hbias.reset()
        gds.hbias.mem = numpy.zeros((1, 1000), dtype=numpy.float64)
        gds.hbias.mem[:] = grad_data["hbias"][:].transpose()
        gds.hbias.initialize(device=NumpyDevice())
        gds.vbias = Array()
        gds.vbias.reset()
        gds.vbias.mem = numpy.zeros((1, 196), dtype=numpy.float64)
        gds.vbias.mem[:] = grad_data["vbias"][:].transpose()
        gds.vbias.initialize(self.device)
        gds.batch_size = grad_data["h1_out"].shape[0]
        gds.initialize(device=NumpyDevice())
        prng.get().seed(1337)
        gds.run()
        diff1 = numpy.sum(numpy.abs(gds.h1.mem - grad_data["h1_out"]))
        diff2 = numpy.sum(numpy.abs(gds.v1.mem - grad_data["v1_out"]))
        self.assertLess(diff1, 1e-12, " total error  is %0.17f" % diff1)
        self.assertLess(diff2, 1e-12, " total error  is %0.17f" % diff2)
        del launcher

    def test_BatchWeights(self):
        # will make initialization for crated Arrays and make map_read
        # and map_write
        bw = rbm.BatchWeights(self.parent)
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "research/MnistRBM/rbm_data",
            "test_batchWeights.mat")
        test_data = scipy.io.loadmat(data_path)
        bw.batch_size = 128
        bw.v = Array()
        bw.h = Array()
        bw.v.mem = test_data["v"]
        bw.h.mem = test_data["h"]
        bw.initialize(device=NumpyDevice())
        bw.run()
        for v in (bw.vbias_batch, bw.hbias_batch, bw.weights_batch):
            v.map_read()
        diff1 = numpy.abs(numpy.sum(bw.vbias_batch.mem -
                                    test_data["vbias_batch"]))
        diff2 = numpy.abs(numpy.sum(bw.hbias_batch.mem -
                                    test_data["hbias_batch"]))
        diff3 = numpy.abs(numpy.sum(bw.weights_batch.mem -
                                    test_data["W_batch"]))
        self.assertLess(diff1, 1e-12, " total error  is %0.17f" % diff1)
        self.assertLess(diff2, 1e-12, " total error  is %0.17f" % diff2)
        self.assertLess(diff3, 1e-12, " total error  is %0.17f" % diff3)

    def test_GradientsCalculator(self):
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "research/MnistRBM/rbm_data",
            "test_makeGrad.mat")
        test_data = scipy.io.loadmat(data_path)
        mg = rbm.GradientsCalculator(self.parent)
        mg.hbias1 = Array()
        mg.vbias1 = Array()
        mg.hbias0 = Array()
        mg.vbias0 = Array()
        mg.weights0 = Array()
        mg.weights1 = Array()
        mg.vbias0.mem = test_data["vbias0"]
        mg.hbias0.mem = test_data["hbias0"]
        mg.vbias1.mem = test_data["vbias1"]
        mg.hbias1.mem = test_data["hbias1"]
        mg.weights0.mem = test_data["W0"]
        mg.weights1.mem = test_data["W1"]
        mg.initialize(device=NumpyDevice())
        mg.run()
        diff1 = numpy.abs(numpy.sum(mg.vbias_grad.mem -
                                    test_data["vbias_grad"]))
        diff2 = numpy.abs(numpy.sum(mg.hbias_grad.mem -
                                    test_data["hbias_grad"]))
        diff3 = numpy.abs(numpy.sum(mg.weights_grad.mem -
                                    test_data["W_grad"]))
        for v in (diff1, diff2, diff3):
            self.assertLess(v, 1e-12, " total error  is %0.17f" % v)

    def test_UpdateWeights(self):
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "research/MnistRBM/rbm_data",
            "test_updateWeights.mat")
        test_data = scipy.io.loadmat(data_path)
        uw = rbm.WeightsUpdater(self.parent, learning_rate=0.001)
        uw.weights = Array()
        uw.weights.mem = test_data["W_old"]
        uw.vbias = Array()
        uw.vbias.mem = test_data["vbias_old"]
        uw.hbias = Array()
        uw.hbias.mem = test_data["hbias_old"]
        uw.weights_grad = Array()
        uw.weights_grad.mem = test_data["W_grad"].transpose()
        uw.vbias_grad = Array()
        uw.vbias_grad.mem = test_data["vbias_grad"]
        uw.hbias_grad = Array()
        uw.hbias_grad.mem = test_data["hbias_grad"]

        uw.initialize(device=NumpyDevice())
        uw.run()
        diff1 = numpy.abs(numpy.sum(uw.vbias.mem -
                                    test_data["vbias_new"]))
        diff2 = numpy.abs(numpy.sum(uw.hbias.mem -
                                    test_data["hbias_new"]))
        diff3 = numpy.abs(numpy.sum(uw.weights.mem -
                                    test_data["W_new"]))
        self.assertLess(diff1, 1e-12, " total error  is %0.17f" % diff1)
        self.assertLess(diff2, 1e-12, " total error  is %0.17f" % diff2)
        self.assertLess(diff3, 1e-12, " total error  is %0.17f" % diff3)


@assign_backend("ocl")
class OpenCLTestRBMUnits(TestRBMUnits):
    pass


@assign_backend("cuda")
class CUDATestRBMUnits(TestRBMUnits):
    pass

if __name__ == "__main__":
    AcceleratedTest.main()
