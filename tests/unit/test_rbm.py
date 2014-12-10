import gc
import logging
import numpy
import os
import scipy.io
import unittest

from veles.dummy import DummyWorkflow
from veles.memory import Vector

import veles.znicz.rbm as rbm


class TestRBMUnits(unittest.TestCase):
    """Tests for uniits used in RBM workflow.
    """
    def setUp(self):
        test_data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'data', 'rbm', 'test_rbm.mat'))
        self.test_data = scipy.io.loadmat(test_data_path)
        self.device = None

    def tearDown(self):
        gc.collect()
        del self.device

    def test_GradientDescentRBM(self):
        """This function creates GradientDescentRBM unit for MNIST task
        and compares result with the output produced function RBM
        from  MATLAB (http://deeplearning.net/tutorial/rbm.html (25.11.14))
        Raises:
            AssertLess: if unit output is wrong.
        """
        gd_unit = rbm.GradientDescentRBM(DummyWorkflow(), learning_rate=1)
        # TODO(d.podoprikhin) learning_rate make the parametr
        gd_unit.output = Vector()
        gd_unit.output.reset()
        gd_unit.output.mem = numpy.zeros((128, 1000),
                                         dtype=numpy.float64)
        gd_unit.output.mem[:] = self.test_data["h0"][:]
        gd_unit.output.initialize(self.device)
        gd_unit.err_output = gd_unit.output
        gd_unit.input = Vector()
        gd_unit.input.reset()
        gd_unit.input.mem = numpy.zeros((128, 196),
                                        dtype=numpy.float64)
        gd_unit.input.mem[:] = self.test_data["v0_bino"][:]
        gd_unit.input.initialize(self.device)

        gd_unit.bias = Vector()
        gd_unit.bias.reset()
        gd_unit.bias.mem = numpy.zeros((1, 1000),
                                       dtype=numpy.float64)
        gd_unit.bias.mem[:] = numpy.transpose(self.test_data["hbias"])[:]
        gd_unit.bias.initialize(self.device)

        gd_unit.vbias = Vector()
        gd_unit.vbias.reset()
        gd_unit.vbias.mem = numpy.zeros((1, 196),
                                        dtype=numpy.float64)
        gd_unit.vbias.mem[:] = numpy.transpose(self.test_data["vbias"])[:]
        gd_unit.vbias.initialize(self.device)

        gd_unit.weights = Vector()
        gd_unit.weights.reset()
        gd_unit.weights.mem = numpy.zeros((1000, 196),
                                          dtype=numpy.float64)
        gd_unit.weights.mem[:] = numpy.transpose(self.test_data["W"])[:]
        gd_unit.weights.initialize(self.device)

        gd_unit.batch_size = 128
        gd_unit.need_err_input = False
        gd_unit.initialize(device=self.device)
        numpy.random.seed(1337)
        gd_unit.run()
        gd_unit.weights.map_write()
        gd_unit.bias.map_write()
        gd_unit.vbias.map_write()
        diff_vbias = numpy.sum(numpy.abs(gd_unit.vbias.mem -
                               numpy.transpose(self.test_data["vbias1"])))
        diff_bias = numpy.sum(numpy.abs(gd_unit.bias.mem -
                              numpy.transpose(self.test_data["hbias1"])))
        diff_weights = numpy.sum(
            numpy.abs(gd_unit.weights.mem -
                      numpy.transpose(self.test_data["W1"])))

        self.assertLess(diff_vbias, 1e-14,
                        " total error vbias is %0.17f" % diff_vbias)

        self.assertLess(diff_bias, 1e-14,
                        " total error bias is %0.17f" % diff_bias)

        self.assertLess(diff_weights, 1e-12,
                        " total error weights is %0.17f" % diff_weights)

    def test_EvaluatorRBM(self):
        """This function creates EvaluatorRBM unit for MNIST task
        and compares result with the output produced function RBM
        from  MATLAB (http://deeplearning.net/tutorial/rbm.html (25.11.14))
        Raises:
            AssertLess: if unit output is wrong.
        """
        evaluator = rbm.EvaluatorRBM(DummyWorkflow())
        evaluator.input = Vector()
        evaluator.input.reset()
        evaluator.input.mem = numpy.zeros((128, 1000),
                                          dtype=numpy.float64)
        numpy.random.seed(1337)
        evaluator.input.mem[:] = self.test_data["h0"][:]
        evaluator.input.initialize(self.device)

        evaluator.output = Vector()
        evaluator.output.reset()
        evaluator.output.mem = numpy.zeros((128, 196),
                                           dtype=numpy.float64)
        evaluator.output.initialize(self.device)

        evaluator.ground_truth = Vector()
        evaluator.ground_truth.reset()
        evaluator.ground_truth.mem = numpy.zeros((128, 196),
                                                 dtype=numpy.float64)
        evaluator.ground_truth.mem[:] = self.test_data["v0_bino"]
        evaluator.ground_truth.initialize(self.device)

        evaluator.bias = Vector()
        evaluator.bias.reset()
        evaluator.bias.mem = numpy.zeros((1, 1000),
                                         dtype=numpy.float64)
        evaluator.bias.mem = numpy.transpose(self.test_data["hbias"])[:]
        evaluator.bias.initialize(self.device)

        evaluator.vbias = Vector()
        evaluator.vbias.reset()
        evaluator.vbias.mem = numpy.zeros((1, 196),
                                          dtype=numpy.float64)
        evaluator.vbias.mem[:] = numpy.transpose(self.test_data["vbias"])[:]
        evaluator.vbias.initialize(self.device)

        evaluator.weights = Vector()
        evaluator.weights.reset()
        evaluator.weights.mem = numpy.zeros((1000, 196),
                                            dtype=numpy.float64)
        evaluator.weights.mem[:] = numpy.transpose(self.test_data["W"])[:]
        evaluator.weights.initialize(self.device)
        evaluator.minibatch_labels = None
        evaluator.max_samples_per_epoch = 1
        evaluator.minibatch_class = 2
        evaluator.class_lengths = (0, 0, 128)
        evaluator.last_minibatch = True
        evaluator.max_minibatch_size = 128
        evaluator.batch_size = 128
        evaluator.max_epochs = 1
        evaluator.initialize(device=self.device)
        evaluator.run()
        diff = numpy.sum(numpy.abs(evaluator.reconstruction_error -
                         self.test_data["rerr"]))
        self.assertLess(diff, 1e-14, " total error  is %0.17f" % diff)

    def test_All2AllRBM(self):
        """This function creates All2AllRBM unit for MNIST task
        and compares result with the output produced function RBM
        from  MATLAB (http://deeplearning.net/tutorial/rbm.html (25.11.14))
        Raises:
            AssertLess: if unit output is wrong.
        """
        a2a = rbm.All2AllRBM(DummyWorkflow(), output_shape=1000,
                             weights_stddev=0.05)
        # add initialize and input
        a2a.input = Vector()
        a2a.input.reset()
        a2a.input.mem = numpy.zeros((128, 196),
                                    dtype=numpy.float64)
        a2a.input.mem[:] = self.test_data["patches"][0: 128]
        a2a.input.initialize(self.device)
        a2a.batch_size = 128
        a2a.initialize(device=self.device)
        a2a.weights.map_write()
        a2a.bias.map_write()
        a2a.vbias.map_write()
        a2a.weights.mem[:] = numpy.transpose(self.test_data["W"])[:]
        a2a.vbias.mem[:] = numpy.transpose(self.test_data["vbias"])[:]
        a2a.bias.mem = numpy.transpose(self.test_data["hbias"])[:]
        numpy.random.seed(1337)
        a2a.run()
        a2a.output.map_read()
        diff = numpy.sum(a2a.output.mem - self.test_data["h0"])
        self.assertLess(diff, 1e-14, " total error  is %0.17f" % diff)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running RBM tests")
    unittest.main()
