#!/usr/bin/python3.3 -O
"""
Created on April 7, 2014

@author: Lyubov Podoynitsina <lyubov.p@samsung.com>
"""


import datetime
import glob
import logging
import numpy
import os
import shutil
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.rnd as rnd
import veles.znicz.samples.mnist_relu as mnist_relu
import veles.tests.dummy_workflow as dummy_workflow


class TestMnistRelu(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def test_mnist_relu(self):
        logging.info("Will test decision, evaluator, image_saver units")
        rnd.default.seed(numpy.fromfile("%s/veles/znicz/samples/seed"
                                        % (root.common.veles_dir),
                                        dtype=numpy.int32, count=1024))
        mnist_dir = os.path.join(root.common.veles_dir,
                                 "veles/znicz/samples/MNIST")

        root.update = {"decision": {"fail_iterations": (0),
                                    "snapshot_prefix": "mnist_relu_test"},
                       "global_alpha": 0.01,
                       "global_lambda": 0.0,
                       "loader": {"minibatch_maxsize": 60},
                       "path_for_load_data_test_images":
                       os.path.join(mnist_dir, "t10k-images.idx3-ubyte"),
                       "path_for_load_data_test_label":
                       os.path.join(mnist_dir, "t10k-labels.idx1-ubyte"),
                       "path_for_load_data_train_images":
                       os.path.join(mnist_dir, "train-images.idx3-ubyte"),
                       "path_for_load_data_train_label":
                       os.path.join(mnist_dir, "train-labels.idx1-ubyte")}

        i = datetime.datetime.now()
        root.image_saver.out = os.path.join(
            root.common.cache_dir, "tmp/test_image_saver_%s" %
            (i.strftime('%Y_%m_%d_%H_%M_%S')))
        root.image_saver.limit = 5
        self.w = mnist_relu.Workflow(dummy_workflow.DummyWorkflow(),
                                     layers=[100, 10], device=self.device)
        self.w.initialize()
        self.w.run()
        files_test = glob.glob("%s/*.png" % (os.path.join(root.image_saver.out,
                                                          "test")))
        logging.info("files in test: %s", files_test)
        logging.info("Number of files in test: %s", len(files_test))
        self.assertGreaterEqual(root.image_saver.limit, len(files_test))
        files_validation = glob.glob("%s/*.png" %
                                     (os.path.join(root.image_saver.out,
                                                   "validation")))
        logging.info("files in validation: %s", files_validation)
        logging.info("Number of files in validation: %s",
                     len(files_validation))
        self.assertGreaterEqual(root.image_saver.limit, len(files_validation))
        files_train = glob.glob("%s/*.png" %
                                (os.path.join(root.image_saver.out, "train")))
        logging.info("files in train: %s", files_train)
        logging.info("Number of files in train: %s", len(files_train))
        self.assertGreaterEqual(root.image_saver.limit, len(files_train))
        for rt, dirs, files in os.walk(root.image_saver.out):
            for f in files:
                os.unlink(os.path.join(rt, f))
            for d in dirs:
                shutil.rmtree(os.path.join(rt, d))
            os.removedirs(root.image_saver.out)
            logging.info("Remove directory %s" % (root.image_saver.out))

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 552)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
