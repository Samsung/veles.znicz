#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on April 7, 2014

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


import datetime
from glob import glob
import logging
import numpy
import os
import shutil
import time
import unittest

from veles.config import root
import veles.memory as formats
import veles.prng as prng
import veles.znicz.image_saver as image_saver
import veles.dummy as dummy_workflow


class TestImageSaver(unittest.TestCase):
    def setUp(self):
        i = datetime.datetime.now()
        root.image_saver.out_dirs = [
            os.path.join(
                root.common.cache_dir, "tmpimg/test_image_saver_%s/test"
                % (i.strftime('%Y_%m_%d_%H_%M_%S'))),
            os.path.join(root.common.cache_dir,
                         "tmpimg/test_image_saver_%s/validation"
                         % (i.strftime('%Y_%m_%d_%H_%M_%S'))),
            os.path.join(root.common.cache_dir,
                         "tmpimg/test_image_saver_%s/train"
                         % (i.strftime('%Y_%m_%d_%H_%M_%S')))]
        self.img_saver_SM = image_saver.ImageSaver(
            dummy_workflow.DummyWorkflow(), out_dirs=root.image_saver.out_dirs)

        self.fill_data()

        self.img_saver_SM.input = self.minibatch_data
        self.img_saver_SM.labels = self.lbls
        self.img_saver_SM.indices = self.indices
        self.img_saver_SM.minibatch_size = 20
        self.img_saver_SM.output = self.output
        self.img_saver_SM.color_space = "RGB"

    def tearDown(self):
        logging.info("All Ok")
        for i in range(0, 2):
            for rt, dirs, files in os.walk(root.image_saver.out_dirs[i]):
                for f in files:
                    os.unlink(os.path.join(rt, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(rt, d))
                os.removedirs(root.image_saver.out_dirs[i])
                logging.info("Remove directory %s" %
                             (root.image_saver.out_dirs[i]))

    def fill_data(self):
        self.minibatch_data = formats.Vector()
        self.minibatch_data.mem = numpy.zeros([20, 32, 32],
                                              dtype=numpy.float32)
        prng.get().fill(self.minibatch_data.mem)

        self.lbls = formats.Vector()
        self.lbls.mem = numpy.array([1, 0, 0, 2, 4,
                                     7, 9, 3, 6, 8,
                                     6, 3, 5, 5, 5,
                                     9, 0, 1, 1, 0], dtype=numpy.int8)

        self.max_idx = formats.Vector()
        self.max_idx.mem = numpy.array([1, 2, 1, 2, 4,
                                        7, 8, 3, 6, 8,
                                        6, 1, 5, 5, 9,
                                        9, 0, 1, 4, 0],
                                       dtype=self.lbls.mem.dtype)

        self.indices = formats.Vector()
        self.indices.mem = numpy.array([0, 1, 2, 3, 4,
                                        5, 6, 7, 8, 9,
                                        0, 1, 2, 3, 4,
                                        5, 6, 7, 8, 9],
                                       dtype=self.lbls.mem.dtype)

        self.output = formats.Vector()
        self.output.mem = numpy.zeros([20, 10], dtype=numpy.float32)
        prng.get().fill(self.output.mem)
        self.output.mem -= self.output.mem.max()
        numpy.exp(self.output.mem, self.output.mem)
        smm = self.output.mem.sum()
        if smm != 0:
            self. output.mem /= smm

    def test_image_saver_MSE_test(self):
        logging.info("Will test image_saver unit for MSE, test")
        i = datetime.datetime.now()
        root.image_saver.out_dirs = [
            os.path.join(
                root.common.cache_dir, "tmpimg/test_image_saver_%s/test"
                % (i.strftime('%Y_%m_%d_%H_%M_%S'))),
            os.path.join(root.common.cache_dir,
                         "tmpimg/test_image_saver_%s/validation"
                         % (i.strftime('%Y_%m_%d_%H_%M_%S'))),
            os.path.join(root.common.cache_dir,
                         "tmpimg/test_image_saver_%s/train"
                         % (i.strftime('%Y_%m_%d_%H_%M_%S')))]
        self.img_saver_MSE = image_saver.ImageSaver(
            dummy_workflow.DummyWorkflow(), out_dirs=root.image_saver.out_dirs)
        self.fill_data()
        self.img_saver_MSE.input = self.minibatch_data
        self.img_saver_MSE.labels = self.lbls
        self.img_saver_MSE.indices = self.indices
        self.img_saver_MSE.minibatch_size = 20
        self.output = formats.Vector()
        self.output.mem = numpy.zeros([20, 32, 32], dtype=numpy.float32)
        self.img_saver_MSE.output = self.output
        self.img_saver_MSE.color_space = "RGB"
        self.img_saver_MSE.this_save_time = time.time()
        self.target = formats.Vector()
        self.target.mem = numpy.zeros([20, 32, 32], dtype=numpy.float32)
        prng.get().fill(self.target.mem)
        self.img_saver_MSE.minibatch_class = 0
        self.img_saver_MSE.target = self.target
        self.img_saver_MSE.initialize()
        self.img_saver_MSE.run()
        files_test = []
        for root_path, _tmp, files in os.walk(
                root.image_saver.out_dirs[0], followlinks=True):
            for f in files:
                f_path = os.path.join(root_path, f)
                files_test.append(f_path)
        logging.info("files in test: %s", files_test)
        logging.info("Number of files in test: %s", len(files_test))
        self.assertEqual(len(files_test), 60)

    def test_image_saver_SM_t(self):
        logging.info("Will test image_saver unit for Softmax, test")
        self.img_saver_SM.this_save_time = time.time()
        self.img_saver_SM.max_idx = self.max_idx
        self.img_saver_SM.minibatch_class = 0
        self.img_saver_SM.initialize()
        self.img_saver_SM.run()
        files_test = glob("%s/*.png" % root.image_saver.out_dirs[0])
        logging.info("files in test: %s", files_test)
        logging.info("Number of files in test: %s", len(files_test))
        self.assertEqual(len(files_test), 6)

    def test_image_saver_SM_validation(self):
        logging.info("Will test image_saver unit for Softmax, validation")
        self.img_saver_SM.this_save_time = time.time()
        self.img_saver_SM.max_idx = self.max_idx
        self.img_saver_SM.minibatch_class = 1
        self.img_saver_SM.initialize()
        self.img_saver_SM.run()

        files_validation = glob("%s/*.png" % root.image_saver.out_dirs[1])
        logging.info("files in validation: %s", files_validation)
        logging.info("Number of files in validation: %s",
                     len(files_validation))
        self.assertEqual(len(files_validation), 6)

    def test_image_saver_SM_train(self):
        logging.info("Will test image_saver unit for Softmax, train")
        self.img_saver_SM.this_save_time = time.time()
        self.img_saver_SM.max_idx = self.max_idx
        self.img_saver_SM.minibatch_class = 2
        self.img_saver_SM.initialize()
        self.img_saver_SM.run()

        files_train = glob("%s/*.png" % (root.image_saver.out_dirs[2]))
        logging.info("files in train: %s", files_train)
        logging.info("Number of files in train: %s", len(files_train))
        self.assertEqual(len(files_train), 6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
