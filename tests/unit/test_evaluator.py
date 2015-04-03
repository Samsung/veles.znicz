#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on November 18, 2013

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

from veles.config import root
import veles.memory as formats
import veles.opencl_types as opencl_types
import veles.prng as random_generator
from veles.tests import AcceleratedTest, assign_backend
import veles.znicz.evaluator as evaluator


class TestEvaluator(AcceleratedTest):
    ABSTRACT = True

    def test_mse(self):
        batch_size = 25
        sample_size = 7500

        dtype = opencl_types.dtypes[root.common.precision_type]
        output = numpy.empty([batch_size, sample_size], dtype=dtype)
        random_generator.get().fill(output)

        target = numpy.empty_like(output)
        random_generator.get().fill(target)

        ev = evaluator.EvaluatorMSE(self.parent)
        ev.output = formats.Vector()
        ev.output.mem = output.copy()
        ev.target = formats.Vector()
        ev.target.mem = target.copy()
        ev.batch_size = batch_size - 5
        gold_err_output = (output - target) / (batch_size - 5)
        gold_err_output[ev.batch_size:] = 0

        ev.initialize(device=self.device)
        ev.err_output.map_invalidate()
        ev.err_output.mem[:] = 1.0e30
        ev.run()

        ev.err_output.map_read()
        max_diff = numpy.fabs(ev.err_output.mem - gold_err_output).max()
        self.info("Difference is %.12f", max_diff)
        self.assertLess(max_diff, 1.0e-4)

    def test_softmax(self):
        batch_size = 25
        n_classes = 75

        dtype = opencl_types.dtypes[root.common.precision_type]
        output = numpy.empty([batch_size, n_classes], dtype=dtype)
        random_generator.get().fill(output)
        max_idx = numpy.empty(batch_size, dtype=numpy.int32)
        for i, sample in enumerate(output):
            max_idx[i] = sample.argmax()
            sample -= sample[max_idx[i]]
            numpy.exp(sample, sample)
            sample /= sample.sum()

        labels = numpy.empty(batch_size, dtype=numpy.int32)
        labels[:] = random_generator.get().randint(0, n_classes, batch_size)

        target = numpy.empty_like(output)
        for i, sample in enumerate(target):
            sample[:] = 0
            sample[labels[i]] = 1

        ev = evaluator.EvaluatorSoftmax(self.parent)
        ev.output = formats.Vector()
        ev.output.mem = output.copy()
        ev.labels = formats.Vector()
        ev.labels.mem = labels.copy()
        ev.max_idx = formats.Vector()
        ev.max_idx.mem = max_idx
        ev.batch_size = batch_size - 5

        gold_err_output = (output - target) / (batch_size - 5)
        gold_err_output[ev.batch_size:] = 0

        ev.initialize(device=self.device)
        ev.err_output.map_invalidate()
        ev.err_output.mem[:] = 1.0e30
        ev.run()

        ev.err_output.map_read()
        max_diff = numpy.fabs(ev.err_output.mem - gold_err_output).max()
        self.info("Difference is %.12f", max_diff)
        self.assertLess(max_diff, 1.0e-4)


@assign_backend("ocl")
class OpenCLTestEvaluator(TestEvaluator):
    pass


@assign_backend("cuda")
class CUDATestEvaluator(TestEvaluator):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
