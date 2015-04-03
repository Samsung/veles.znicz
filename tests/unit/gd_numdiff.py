# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 26, 2014

Helper class for numeric differentiation tests.

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
import veles.memory as formats
import veles.znicz.all2all as all2all


class GDNumDiff(object):
    """Helper class for numeric differentiation tests.

    WARNING: it is invalid for single precision float data type.
    """
    def numdiff_check(self, forward, vector_to_check, vector_value_map,
                      vector_output, target, derivative_to_check,
                      logging_info, assertLess, error_function, batch_size,
                      limit=None, threshold=1.0e-5):
        """Checks derivative by numeric differentiation based on MSE to target.

        Parameters:
            forward: forward unit instance, should have input attribute
                     of type Vector where input.mem.shape[0] is the batch size.
            vector_to_check: vector on which to do numeric differentiation.
            vector_value_map: dictionary of vectors to set => its values,
                              should contain vector_to_check.
            vector_output: output vector to compute MSE based on target.
            target: target numpy array for MSE criteria.
            derivative: numpy array of derivative to check
                        (computed by backward unit).
            logging_info: pointer to logging function
                          (logging.info for example).
            assertLess: pointer to assertLess function.
        """
        for v in vector_value_map.keys():
            if v is None or v.mem is None:
                continue
            if v.dtype == numpy.float32:
                logging_info(
                    "numdiff_check is invalid for single precision "
                    "float data type, will skip it")
                return

        numdiff = formats.NumDiff()

        mem = formats.ravel(vector_to_check.mem)
        derivative_to_check = derivative_to_check.ravel()
        for offs in range(mem.shape[0]):
            for i, p in enumerate(numdiff.points):
                for k, v in vector_value_map.items():
                    if v is None or k is None or k.mem is None:
                        continue
                    k.map_invalidate()
                    formats.ravel(k.mem)[:] = v.ravel()[:]
                mem[offs] = mem[offs] + p
                forward.stopped = False
                forward.run()
                vector_output.map_read()
                numdiff.errs[i] = error_function(vector_output.mem.ravel(),
                                                 target.ravel(), batch_size)
            derivative = numdiff.derivative
            d = numpy.fabs(derivative - derivative_to_check[offs])
            logging_info("%.2e %.2e %.2e" %
                         (derivative, derivative_to_check[offs], d))
            assertLess(d, threshold, "Numeric diff test failed")
            if limit is not None and offs >= limit - 1:
                logging_info("Limit of %d checks reached, skipping the rest",
                             limit)
                return

    @staticmethod
    def sse(y, t, batch_size):
        return numpy.square(y - t).sum() * 0.5

    @staticmethod
    def mse(y, t, batch_size):
        return numpy.square(y - t).sum() * 0.5 / batch_size

    @staticmethod
    def cross_entropy_sum(y, t, batch_size):
        idx = numpy.nonzero(t)
        return (t[idx] * numpy.log(t[idx] / y[idx])).sum()

    @staticmethod
    def cross_entropy_mean(y, t, batch_size):
        idx = numpy.nonzero(t)
        return (t[idx] * numpy.log(t[idx] / y[idx])).sum() / batch_size

    def numdiff_check_gd(self, forward, inp, weights, bias, target,
                         err_input, weights_derivative, bias_derivative,
                         logging_info, assertLess,
                         error_function_averaged=True,
                         limit=None, threshold=1.0e-5):
        """Tests all derivatives for a typical gradient descent unit.
        """
        if error_function_averaged:
            ef = (
                GDNumDiff.cross_entropy_mean
                if isinstance(forward, all2all.All2AllSoftmax)
                else GDNumDiff.mse)
        else:
            ef = (
                GDNumDiff.cross_entropy_sum
                if isinstance(forward, all2all.All2AllSoftmax)
                else GDNumDiff.sse)
        batch_size = inp.shape[0]
        for vector, derivative, nme in (
                (forward.input, err_input, "err_input"),
                (forward.weights, weights_derivative, "weights"),
                (getattr(forward, "bias", None), bias_derivative, "bias")):
            if derivative is None:
                continue
            logging_info("Checking %s via numeric differentiation on %s",
                         nme, forward.__class__.__name__)
            self.numdiff_check(
                forward, vector, {forward.input: inp,
                                  forward.weights: weights,
                                  getattr(forward, "bias", None): bias},
                forward.output, target, derivative,
                logging_info, assertLess, ef, batch_size,
                limit=limit, threshold=threshold)
