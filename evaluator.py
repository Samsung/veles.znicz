"""
Created on Apr 1, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import numpy
from zope.interface import implementer

from veles.distributable import TriviallyDistributable
import veles.error as error
import veles.formats as formats
from veles.opencl_units import OpenCLUnit, IOpenCLUnit


class EvaluatorBase(OpenCLUnit):
    """Base class for Evaluators.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "EVALUATOR")
        kwargs["error_function_averaged"] = kwargs.get(
            "error_function_averaged", True)
        super(EvaluatorBase, self).__init__(workflow, **kwargs)
        self.error_function_averaged = kwargs["error_function_averaged"]
        self.output = None  # formats.Vector()
        self.err_output = formats.Vector()
        self.batch_size = 0
        self.krn_constants_i_ = None
        self.krn_constants_f_ = None
        self.demand("output", "batch_size")

    def initialize(self, device, **kwargs):
        super(EvaluatorBase, self).initialize(device, **kwargs)

        dtype = self.output.mem.dtype

        self.krn_constants_i_ = numpy.zeros(1, dtype=numpy.int32)
        self.krn_constants_f_ = numpy.zeros(1, dtype=dtype)

        if (self.err_output.mem is None or
                self.err_output.mem.size != self.output.mem.size):
            self.err_output.reset()
            self.err_output.mem = numpy.zeros(self.output.mem.shape,
                                              dtype=dtype)

        self.output.initialize(self)
        self.err_output.initialize(self)


@implementer(IOpenCLUnit)
class EvaluatorSoftmax(EvaluatorBase, TriviallyDistributable):
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
        compute_confusion_matrix = kwargs.get("compute_confusion_matrix", True)
        kwargs["compute_confusion_matrix"] = compute_confusion_matrix
        super(EvaluatorSoftmax, self).__init__(workflow, **kwargs)
        self.labels = None  # formats.Vector()
        self.compute_confusion_matrix = compute_confusion_matrix
        self.confusion_matrix = formats.Vector()
        self.n_err = formats.Vector()
        self.max_idx = None  # formats.Vector()
        self.max_err_output_sum = formats.Vector()
        self.demand("labels", "max_idx")

    def initialize(self, device, **kwargs):
        super(EvaluatorSoftmax, self).initialize(device=device, **kwargs)
        self.cl_sources_["evaluator.cl"] = {}

        dtype = self.output.mem.dtype

        if self.n_err.mem is None:
            self.n_err.reset()
            self.n_err.mem = numpy.zeros(1, dtype=numpy.int32)
        else:
            assert self.n_err.mem.size == 1

        out_size = self.output.mem.size // self.output.mem.shape[0]
        if self.compute_confusion_matrix:
            if (self.confusion_matrix.mem is None or
                    self.confusion_matrix.mem.size != out_size * out_size):
                self.confusion_matrix.reset()
                self.confusion_matrix.mem = numpy.zeros(
                    [out_size, out_size], dtype=numpy.int32)
        else:
            self.confusion_matrix.reset()

        if (self.max_err_output_sum.mem is None or
                self.max_err_output_sum.mem.size < 1):
            self.max_err_output_sum.reset()
            self.max_err_output_sum.mem = numpy.zeros(1, dtype=dtype)

        self.confusion_matrix.initialize(self)
        self.n_err.initialize(self)
        self.max_idx.initialize(self)
        self.labels.initialize(self)
        self.max_err_output_sum.initialize(self)

        if self.device is not None:
            EvaluatorSoftmax.ocl_init(self, device)

    def ocl_init(self, device):
        dtype = self.output.mem.dtype
        block_size = min(self.err_output.shape[0], 128)
        defines = {
            "BLOCK_SIZE": block_size,
            "BATCH": self.err_output.shape[0],
            "Y": self.err_output.sample_size
        }
        self._local_size = [block_size]
        self._global_size = self._local_size

        self.build_program(defines, "ev_%d.cl" % self.output.sample_size,
                           dtype=dtype)

        self.assign_kernel("ev_sm")
        self.set_args(self.output, self.max_idx, self.labels,
                      self.err_output, self.n_err, self.confusion_matrix,
                      self.max_err_output_sum)

    def ocl_run(self):
        self.err_output.unmap()
        self.output.unmap()
        self.max_idx.unmap()
        self.labels.unmap()
        self.n_err.unmap()
        self.confusion_matrix.unmap()
        self.max_err_output_sum.unmap()

        self.krn_constants_i_[0] = self.batch_size
        self.set_arg(7, self.krn_constants_i_[0:1])
        self.krn_constants_f_[0] = (
            1.0 / self.batch_size if self.error_function_averaged else 1.0)
        self.set_arg(8, self.krn_constants_f_[0:1])

        self.execute_kernel(self._global_size, self._local_size)

    def cpu_run(self):
        self.err_output.map_invalidate()
        self.output.map_read()
        self.max_idx.map_read()
        self.labels.map_read()
        self.n_err.map_write()
        self.confusion_matrix.map_write()
        self.max_err_output_sum.map_write()

        batch_size = self.batch_size
        labels = self.labels.mem
        confusion_matrix = self.confusion_matrix.mem

        n_ok = 0
        multiplier = 1.0 / batch_size if self.error_function_averaged else 1.0
        for i in range(batch_size):  # loop by batch
            output = formats.ravel(self.output[i])
            err_output = formats.ravel(self.err_output[i])

            max_idx = self.max_idx[i]
            confusion_matrix[max_idx, labels[i]] += 1
            if max_idx == labels[i]:
                n_ok += 1

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


@implementer(IOpenCLUnit)
class EvaluatorMSE(EvaluatorBase, TriviallyDistributable):
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
        labels: labels for a Batch (may be None).
        class_targets: target for each class (may be None).
        n_err: number of wrong recognized samples
            (if labels and class_targets is not None).
    """
    def __init__(self, workflow, **kwargs):
        super(EvaluatorMSE, self).__init__(workflow, **kwargs)
        self.target = None  # formats.Vector()
        self.metrics = formats.Vector()
        self.mse = formats.Vector()
        self.labels = None
        self.class_targets = None
        self.n_err = formats.Vector()
        self.demand("target")

    def initialize(self, device, **kwargs):
        super(EvaluatorMSE, self).initialize(device=device, **kwargs)

        if self.target.shape != self.output.shape:
            raise error.BadFormatError("target.shape != output.shape")

        self.cl_sources_["evaluator.cl"] = {}

        dtype = self.output.mem.dtype

        if self.metrics.mem is None or self.metrics.mem.size < 3:
            self.metrics.reset()
            self.metrics.mem = numpy.zeros(3, dtype=dtype)
            self.metrics[2] = 1.0e30  # mse_min

        if (self.mse.mem is None or
                self.mse.mem.size != self.err_output.mem.shape[0]):
            self.mse.reset()
            self.mse.mem = numpy.zeros(self.err_output.mem.shape[0],
                                       dtype=dtype)

        if self.labels is not None and self.class_targets is not None:
            self.n_err.reset()
            self.n_err.mem = numpy.zeros(2, dtype=numpy.int32)
            self.cl_sources_["mse_find_closest.cl"] = {}
            self.class_targets.initialize(self)
            self.labels.initialize(self)
            self.n_err.initialize(self)

        self.target.initialize(self)
        self.metrics.initialize(self)
        self.mse.initialize(self)

        if self.device is not None:
            EvaluatorMSE.ocl_init(self, device)

    def ocl_init(self, device):
        dtype = self.output.mem.dtype
        block_size = min(self.err_output.shape[0], 128)
        defines = {
            'BLOCK_SIZE': block_size,
            'BATCH': self.err_output.shape[0],
            'Y': self.err_output.sample_size,
            'SAMPLE_SIZE': 'Y',
            'N_TARGETS': (self.class_targets.shape[0]
                          if self.class_targets is not None else 0)}
        self._local_size = [block_size]
        self._global_size = self._local_size

        self.build_program(defines, "ev_%d.cl" % self.output.sample_size,
                           dtype=dtype)

        self.assign_kernel("ev_mse")
        self.set_args(self.output, self.target, self.err_output,
                      self.metrics, self.mse.devmem)

        if self.labels is not None and self.class_targets is not None:
            self.krn_find_closest_ = self.get_kernel("mse_find_closest")
            self.krn_find_closest_.set_args(
                self.output.devmem,
                self.class_targets.devmem,
                self.labels.devmem,
                self.n_err.devmem)

    def ocl_run(self):
        self.err_output.unmap()
        self.output.unmap()
        self.target.unmap()
        self.metrics.unmap()
        self.mse.unmap()

        batch_size = self.batch_size
        self.krn_constants_i_[0] = batch_size
        self.set_arg(5, self.krn_constants_i_[0:1])
        self.krn_constants_f_[0] = (
            1.0 / self.batch_size if self.error_function_averaged else 1.0)
        self.set_arg(6, self.krn_constants_f_[0:1])

        self.execute_kernel(self._global_size, self._local_size)

        if self.labels is not None and self.class_targets is not None:
            self.class_targets.unmap()
            self.labels.unmap()
            self.n_err.unmap()
            self.execute_kernel([batch_size], None, self.krn_find_closest_)

    def cpu_run(self):
        self.output.map_read()
        self.target.map_read()
        self.metrics.map_write()
        self.err_output.map_invalidate()
        self.mse.map_invalidate()

        assert(self.output.shape == self.target.shape == self.err_output.shape)
        batch_size = self.batch_size
        err_output = self.err_output.matrix[:batch_size]
        output = self.output.matrix[:batch_size]
        target = self.target.matrix[:batch_size]
        mse = self.mse.mem[:batch_size]

        err_output[:] = output - target
        if self.error_function_averaged:
            err_output *= 1.0 / batch_size
        self.err_output.mem[batch_size:] = 0
        mse[:] = numpy.sqrt(numpy.square(err_output).sum(axis=1) /
                            err_output.shape[1])
        self.mse.mem[batch_size:] = 0

        self.metrics.mem[0] += numpy.sum(self.mse.mem)
        self.metrics.mem[1] = max(self.metrics.mem[1], self.mse.mem.max())
        self.metrics.mem[2] = min(self.metrics.mem[2], self.mse.mem.min())

        if self.labels is not None and self.class_targets is not None:
            raise NotImplementedError(
                "CPU code for calculating number of errors in case of MSE "
                "is not implemented.")
