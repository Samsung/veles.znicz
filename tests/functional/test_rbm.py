#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on November 6, 2014

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

from veles.config import root
from veles.tests import multi_device
import veles.prng as prng
from veles.znicz.tests.functional import StandardTest
from veles.znicz.tests.research.MnistRBM.mnist_rbm import MnistRBMWorkflow


root.mnist_rbm.update({
    "all2all": {"weights_stddev": 0.05, "output_sample_shape": 1000},
    "decision": {"max_epochs": 2},
    "snapshotter": {"prefix": "mnist_rbm"},
    "loader": {"minibatch_size": 128, "force_numpy": True,
               "data_path":
               os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "research/MnistRBM/rbm_data",
                            "test_rbm_functional.mat")}})


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
            os.path.dirname(__file__), "..", "research/MnistRBM/rbm_data",
            'R_141014_init.mat'))
        learned_weights = scipy.io.loadmat(
            os.path.join(os.path.dirname(__file__),
                         "..", "research/MnistRBM/rbm_data",
                         'R_141014_learned.mat'))
        self.info("MNIST RBM TEST")
        workflow = MnistRBMWorkflow(self.parent)
        workflow.initialize(device=self.device, snapshot=False)
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
