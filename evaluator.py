# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 1, 2013

Defines units which evaluate the target quality function during the neural
network training.

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


from __future__ import division

import numpy
import six
from zope.interface import implementer

from veles.distributable import TriviallyDistributable, IDistributable
import veles.error as error
from veles.loader import TEST
from veles.memory import assert_addr, ravel, Array
from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit, \
    INumpyUnit
from veles.normalization import NoneNormalizer
from veles.opencl_types import numpy_dtype_to_opencl
from veles.result_provider import IResultProvider
from veles.unit_registry import MappedUnitRegistry
from veles.units import Unit, UnitCommandLineArgumentsRegistry


class EvaluatorsRegistry(UnitCommandLineArgumentsRegistry,
                         MappedUnitRegistry):
    mapping = "evaluators"
    base = Unit
    loss_mapping = {}

    def __init__(cls, name, bases, clsdict):
        super(EvaluatorsRegistry, cls).__init__(name, bases, clsdict)
        if "LOSS" in clsdict and "MAPPING" in clsdict:
            EvaluatorsRegistry.loss_mapping[clsdict[
                "LOSS"]] = clsdict["MAPPING"]


@implementer(IResultProvider, IDistributable)
@six.add_metaclass(EvaluatorsRegistry)
class EvaluatorBase(AcceleratedUnit, TriviallyDistributable):
    hide_from_registry = True
    """Base class for evaluators.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "EVALUATOR")
        super(EvaluatorBase, self).__init__(workflow, **kwargs)
        self.mean = kwargs.get("mean", True)
        self.err_output = Array()
        self._merged_output = Array()
        self.krn_constants_i_ = None
        self.krn_constants_f_ = None
        self.demand("output", "batch_size")
        if self.testing:
            self.demand("class_lengths", "offset")

    @property
    def mean(self):
        """
        :return: True if the error function averages values. Default is True.
        """
        return self._mean

    @mean.setter
    def mean(self, value):
        if not isinstance(value, bool):
            raise TypeError("mean must be boolean (got %s)" % type(value))
        self._mean = value

    @property
    def merged_output(self):
        assert self.testing
        return self._merged_output.mem

    def initialize(self, device, **kwargs):
        super(EvaluatorBase, self).initialize(device, **kwargs)
        dtype = self.output.dtype
        if self.testing:
            self._merged_output.reset(numpy.zeros(
                (self.class_lengths[TEST],) + self.output.shape[1:], dtype))
            return

        self.krn_constants_i_ = numpy.zeros(1, numpy.int32)
        self.krn_constants_f_ = numpy.zeros(1, dtype)
        self.err_output.reset(numpy.zeros_like(self.output.mem, dtype))

        for vec in self.output, self.err_output:
            vec.initialize(self.device)

    def run(self):
        if self.testing:
            self.output.map_read()
            self.merge_output()
            return
        return super(EvaluatorBase, self).run()

    def merge_output(self):
        self.merged_output[self.offset - self.batch_size:self.offset] = \
            self.output[:self.batch_size]

    def get_metric_names(self):
        if self.testing:
            return {"Output"}
        return set()

    def get_metric_values(self):
        if self.testing:
            return {"Output": self.merged_output}
        return {}


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class EvaluatorSoftmax(EvaluatorBase):

    MAPPING = "evaluator_softmax"
    LOSS = "softmax"

    """Evaluator for nn softmax output from the batch labels.

    Must be assigned before initialize():
        output
        labels
        batch_size
        max_idx

    Updates after run():
        err_output
        n_err
        confusion_matrix
        max_err_output_sum

    Creates within initialize():
        err_output
        n_err
        confusion_matrix
        max_err_output_sum

    Attributes:
        labels: labels for Batch.
        output: output of the network_common as Batch.
        err_output: backpropagation errors based on labels.
        batch_size: number of elements in output to evaluate.
        confusion_matrix: confusion matrix for the output.
        compute_confusion_matrix: compute confusion matrix or not.
        max_idx: indexes of element with maximum real value for each sample.
        max_err_output_sum: maximum of backpropagated error sum by sample.
    """
    def __init__(self, workflow, **kwargs):
        super(EvaluatorSoftmax, self).__init__(workflow, **kwargs)
        self.compute_confusion_matrix = kwargs.get(
            "compute_confusion_matrix", True)
        self.confusion_matrix = Array()
        self.n_err = Array()
        self.max_err_output_sum = Array()
        self.class_keys = None
        self.demand("labels", "max_idx")
        if self.testing:
            self.demand("labels_mapping")

    def initialize(self, device, **kwargs):
        super(EvaluatorSoftmax, self).initialize(device=device, **kwargs)
        if self.testing:
            return
        self.sources_["evaluator"] = {}

        dtype = self.output.dtype

        if not self.n_err:
            self.n_err.reset(numpy.zeros(2, dtype=numpy.int32))
        else:
            assert self.n_err.size == 2

        out_size = self.output.sample_size
        if self.compute_confusion_matrix:
            if not self.confusion_matrix:
                self.confusion_matrix.reset(
                    numpy.zeros([out_size, out_size], numpy.int32))
            else:
                assert self.confusion_matrix.size == out_size * out_size
        else:
            self.confusion_matrix.reset()

        if not self.max_err_output_sum:
            self.max_err_output_sum.reset(numpy.zeros(1, dtype))
        else:
            assert self.max_err_output_sum.size == 1

        self.init_vectors(self.confusion_matrix, self.n_err, self.max_idx,
                          self.labels, self.max_err_output_sum)

    def _gpu_init(self):
        dtype = self.output.dtype
        block_size = min(self.err_output.shape[0], 256)
        self.build_program(
            cache_file_name="%s_%d_%d" % (self.__class__.__name__,
                                          self.output.shape[0],
                                          self.output.sample_size),
            dtype=dtype, block_size=block_size,
            max_batch_size=self.err_output.shape[0],
            output_size=self.err_output.sample_size)
        self.assign_kernel("evaluate_softmax")
        self.set_args(self.output, self.max_idx, self.labels,
                      self.skip_args(2), self.n_err, self.confusion_matrix,
                      self.max_err_output_sum, self.err_output)
        return block_size

    def ocl_init(self):
        if self.testing:
            return
        block_size = self._gpu_init()
        self._global_size = [block_size]
        self._local_size = [block_size]

    def cuda_init(self):
        if self.testing:
            return
        block_size = self._gpu_init()
        self._global_size = (1, 1, 1)
        self._local_size = (block_size, 1, 1)

    def _gpu_run(self):
        self.unmap_vectors(
            self.err_output, self.output, self.max_idx, self.labels,
            self.n_err, self.confusion_matrix, self.max_err_output_sum)

        self.krn_constants_i_[0] = self.batch_size
        self.set_arg(3, self.krn_constants_i_[0:1])
        self.krn_constants_f_[0] = 1.0 / self.batch_size if self.mean else 1.0
        self.set_arg(4, self.krn_constants_f_[0:1])

        self.execute_kernel(self._global_size, self._local_size)

    def ocl_run(self):
        return self._gpu_run()

    def cuda_run(self):
        return self._gpu_run()

    def numpy_run(self):
        self.err_output.map_invalidate()
        for vec in self.output, self.max_idx, self.labels:
            vec.map_read()
        for vec in self.n_err, self.confusion_matrix, self.max_err_output_sum:
            vec.map_write()

        batch_size = self.batch_size
        labels = self.labels.mem
        confusion_matrix = self.confusion_matrix.mem

        n_ok = 0
        n_total = 0
        multiplier = 1.0 / batch_size if self.mean else 1.0
        for i in range(batch_size):  # loop by batch
            if labels[i] < 0:
                self.err_output.mem[i] = 0.0
                continue
            output = ravel(self.output[i])
            err_output = ravel(self.err_output[i])

            max_idx = self.max_idx[i]
            confusion_matrix[max_idx, labels[i]] += 1
            if max_idx == labels[i]:
                n_ok += 1
            n_total += 1

            # Compute softmax output error gradient
            err_output[:] = output[:]
            err_output[labels[i]] -= 1.0
            err_output *= multiplier
            if err_output.dtype in (numpy.complex64, numpy.complex128):
                self.max_err_output_sum[0] = max(
                    self.max_err_output_sum[0], numpy.linalg.norm(err_output))
            else:
                self.max_err_output_sum[0] = max(
                    self.max_err_output_sum[0], (numpy.fabs(err_output)).sum())
        # Set errors for excessive samples to zero
        if batch_size < self.err_output.mem.shape[0]:
            self.err_output.mem[batch_size:] = 0.0
        self.n_err[0] += batch_size - n_ok
        self.n_err[1] += n_total

    def get_metric_values(self):
        if self.testing:
            output_labels = {}
            class_keys = getattr(self, "class_keys", None)
            for index, labels in enumerate(self.merged_output[:]):
                max_value = 0
                for label_index, value in enumerate(labels):
                    if value >= max_value:
                        max_value = value
                        max_index = label_index
                if class_keys is not None:
                    output_labels[self.class_keys[TEST][
                        index]] = self.labels_mapping[max_index]
                else:
                    output_labels[index] = self.labels_mapping[max_index]
            return {"Output": output_labels}
        return {}


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class EvaluatorMSE(EvaluatorBase):

    MAPPING = "evaluator_mse"
    LOSS = "mse"

    """Evaluator for nn softmax output from the batch labels.

    Must be assigned before initialize():
        output
        target
        batch_size
        labels (may be None)
        class_targets (may be None)

    Updates after run():
        err_output
        confusion_matrix
        max_err_output_sum
        n_err (only if labels and class_targets is not None)

    Creates within initialize():
        err_output
        n_err (only if labels and class_targets is not None)
        max_err_output_sum

    Attributes:
        output: output of the network_common as Batch.
        target: target for the current Batch.
        err_output: backpropagation errors.
        batch_size: number of elements in output to evaluate.
        metrics: [0] - sum of sample's mse, [1] - max of sample's mse,
                 [2] - min of sample's mse.
        mse: array of mse for each sample in minibatch.
        krn_constants_i_: numpy array for constant arguments to kernel.
        labels: labels for a batch (may be None).
        class_targets: target for each class (may be None).
        n_err: number of wrongly recognized samples
            (if labels and class_targets is not None).
    """
    def __init__(self, workflow, **kwargs):
        super(EvaluatorMSE, self).__init__(workflow, **kwargs)
        self.metrics = Array()
        self.mse = Array()
        self.labels = None
        self.class_targets = None
        self.n_err = Array()
        self.root = kwargs.get("root", True)
        self.demand("target", "normalizer")

    @property
    def root(self):
        """
        :return: True if error metric is RMSE, otherwise, MSE (mean sum of
        squares). Default is True.
        """
        return self._root

    @root.setter
    def root(self, value):
        if not isinstance(value, bool):
            raise TypeError("root must be boolean (got %s)" % type(value))
        self._root = value

    def initialize(self, device, **kwargs):
        super(EvaluatorMSE, self).initialize(device=device, **kwargs)
        if self.testing:
            return

        if self.target.size != self.output.size:
            raise error.BadFormatError(
                "target.size != output.size (%s != %s)" %
                (self.target.size, self.output.size))

        self.sources_["evaluator_mse"] = {}
        self.sources_["denormalization"] = {}

        dtype = self.output.dtype

        self.metrics.reset(numpy.zeros(3, dtype=dtype))
        self.metrics[2] = 1.0e30  # mse_min
        self.mse.reset(numpy.zeros(self.err_output.mem.shape[0], dtype))
        self.n_err.reset(numpy.zeros(2, dtype=numpy.int32))
        self.init_vectors(self.n_err, self.target, self.metrics, self.mse)
        if self.class_targets:
            self.class_targets.initialize(self.device)

    def _gpu_init(self):
        dtype = self.output.dtype
        block_size = min(self.err_output.shape[0], 128)
        if self.class_targets:
            self.sources_["mse_find_closest"] = {
                "target_dtype": numpy_dtype_to_opencl(self.class_targets.dtype)
            }

        self.build_program(
            cache_file_name="%s_%d_%d" % (self.__class__.__name__,
                                          self.output.shape[0],
                                          self.output.sample_size),
            dtype=dtype, max_batch_size=self.err_output.shape[0],
            block_size=block_size, output_size=self.err_output.sample_size,
            root=self.root, normalization=self.normalizer.MAPPING,
            targets_number=self.class_targets.shape[0] if self.class_targets
            else None, coeffs=self.normalizer.coefficients)

        self.assign_kernel("evaluate_mse")
        self.set_args(self.output, self.target, self.skip_args(2),
                      self.metrics, self.mse.devmem, self.err_output)

        if self.labels and self.class_targets:
            assert(self.labels.dtype == self.n_err.dtype == numpy.int32)
            self.krn_find_closest_ = self.get_kernel("mse_find_closest")
            self.krn_find_closest_.set_args(
                self.output.devmem,
                self.class_targets.devmem,
                self.labels.devmem,
                self.n_err.devmem)

        return block_size

    def ocl_init(self):
        if self.testing:
            return
        block_size = self._gpu_init()
        self._local_size = [block_size]
        self._global_size = self._local_size
        self._global_size_find_closest_ = lambda: (self.batch_size,)
        self._local_size_find_closest = None

    def cuda_init(self):
        if self.testing:
            return
        block_size = self._gpu_init()
        self._local_size = (block_size, 1, 1)
        self._global_size = (1, 1, 1)
        self._global_size_find_closest_ = lambda: (self.batch_size, 1, 1)
        self._local_size_find_closest = (1, 1, 1)

    def _gpu_run(self):
        self.unmap_vectors(self.err_output, self.output, self.target,
                           self.metrics, self.mse)

        batch_size = self.batch_size
        self.krn_constants_i_[0] = batch_size
        self.set_arg(2, self.krn_constants_i_[0:1])
        self.krn_constants_f_[0] = 1.0 / self.batch_size if self.mean else 1.0
        self.set_arg(3, self.krn_constants_f_[0:1])

        self.execute_kernel(self._global_size, self._local_size)

        if self.labels and self.class_targets:
            self.unmap_vectors(self.class_targets, self.labels, self.n_err)
            self.execute_kernel(self._global_size_find_closest_(),
                                self._local_size_find_closest,
                                self.krn_find_closest_)
            self.n_err.map_write()
            self.n_err.mem[1] += batch_size

    def ocl_run(self):
        return self._gpu_run()

    def cuda_run(self):
        return self._gpu_run()

    def numpy_run(self):
        self.output.map_read()
        self.target.map_read()
        self.metrics.map_write()
        self.err_output.map_invalidate()
        self.mse.map_invalidate()

        assert(self.output.size == self.target.size == self.err_output.size)
        batch_size = self.batch_size
        err_output = self.err_output.matrix[:batch_size]
        assert_addr(err_output, self.err_output.mem)
        output = self.output.matrix[:batch_size]
        assert_addr(output, self.output.mem)
        target = self.target.matrix[:batch_size]
        assert_addr(target, self.target.mem)
        mse = self.mse.mem[:batch_size]
        assert_addr(mse, self.mse.mem)

        err_output[:] = output - target
        if not isinstance(self.normalizer, NoneNormalizer):
            output_copy = output.copy()
            target_copy = target.copy()
            self.normalizer.denormalize(output_copy)
            self.normalizer.denormalize(target_copy)
            denormed_err_output = output_copy - target_copy
        else:
            denormed_err_output = err_output
        self.err_output.mem[batch_size:] = 0
        mse[:] = numpy.square(denormed_err_output).sum(axis=1) / \
            denormed_err_output.shape[1]
        if self.mean:
            err_output /= batch_size
        if self.root:
            numpy.sqrt(mse, mse)
        self.mse.mem[batch_size:] = 0

        self.metrics.mem[0] += mse.sum()
        self.metrics.mem[1] = max(self.metrics.mem[1], mse.max())
        self.metrics.mem[2] = min(self.metrics.mem[2], mse.min())

        if self.labels and self.class_targets:
            self.class_targets.map_read()
            self.labels.map_read()
            self.n_err.map_write()
            class_targets = self.class_targets.matrix
            labels = self.labels.mem
            for i, sample in enumerate(output):
                lbl = numpy.linalg.norm(class_targets - sample,
                                        axis=1).argmin()
                if lbl != labels[i]:
                    self.n_err.mem[0] += 1
                self.n_err.mem[1] += 1

    def merge_output(self):
        if not isinstance(self.normalizer, NoneNormalizer):
            output = self.output[:self.batch_size].copy()
            self.normalizer.denormalize(output)
        else:
            output = self.output.mem
        self.merged_output[self.offset - self.batch_size:self.offset] = output
